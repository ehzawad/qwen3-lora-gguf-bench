from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))

import bench_external  # noqa: E402


def run(*args: str, cwd: Path | None = None, env: dict[str, str] | None = None):
    return subprocess.run(
        list(args),
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def valid_row(concurrency: int = 1, measured: int = 2) -> dict[str, object]:
    expected = concurrency * measured
    tokens = expected * 256
    return {
        "bench_version": "test",
        "tag": f"c{concurrency:03d}",
        "concurrency": concurrency,
        "measured_per_worker": measured,
        "requests_ok": expected,
        "requests_failed": 0,
        "completion_tokens_total": tokens,
        "server_predicted_tokens_delta": float(tokens),
        "output_tokens_per_s": 100.0,
        "output_tokens_per_min": 6000.0,
        "ttft_s": {"p50": 0.1, "p95": 0.2},
        "latency_s": {"p50": 2.0, "p95": 2.2},
        "telemetry": {},
        "ok": True,
    }


class IntegrityGateTests(unittest.TestCase):
    def test_assertion_based_scripts_refuse_optimized_python(self) -> None:
        for name in ("merge.py", "verify_artifacts.py"):
            with self.subTest(script=name):
                completed = run(sys.executable, "-O", str(SCRIPTS / name))
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("optimized Python", completed.stdout + completed.stderr)

    def test_download_hash_verification_survives_optimization(self) -> None:
        path = SCRIPTS / "01_download.py"
        spec = importlib.util.spec_from_file_location("download_script", path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "adapter.safetensors"
            artifact.write_bytes(b"known adapter bytes")
            expected = hashlib.sha256(artifact.read_bytes()).hexdigest()
            self.assertEqual(module.verify_sha256(str(artifact), expected), expected)
            with self.assertRaisesRegex(RuntimeError, "SHA mismatch"):
                module.verify_sha256(str(artifact), "0" * 64)


class ReportTests(unittest.TestCase):
    def invoke_report(self, run_dir: Path):
        return run(sys.executable, str(SCRIPTS / "report.py"), str(run_dir), cwd=REPO)

    def test_empty_run_fails_without_creating_plausible_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            completed = self.invoke_report(run_dir)
            self.assertEqual(completed.returncode, 2)
            self.assertIn("no benchmark-*.json", completed.stderr)
            self.assertFalse((run_dir / "summary.md").exists())

    def test_failed_point_is_labeled_and_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            row = valid_row(concurrency=24, measured=10)
            row.update({
                "requests_ok": 239,
                "requests_failed": 1,
                "completion_tokens_total": 239 * 256,
                "server_predicted_tokens_delta": float(240 * 256),
                "ok": False,
            })
            (run_dir / "benchmark-c024.json").write_text(json.dumps(row), encoding="utf-8")

            completed = self.invoke_report(run_dir)
            self.assertEqual(completed.returncode, 2)
            markdown = (run_dir / "summary.md").read_text(encoding="utf-8")
            self.assertIn("Run status: **FAILED / PARTIAL**", markdown)
            self.assertIn("| 24 | FAIL |", markdown)
            self.assertNotIn("## Throughput scaling", markdown)
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertFalse(summary[0]["ok"])
            self.assertTrue(summary[0]["failure_reasons"])

    def test_valid_run_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            row = valid_row()
            (run_dir / "benchmark-c001.json").write_text(json.dumps(row), encoding="utf-8")
            (run_dir / "experiment.json").write_text(
                json.dumps({"benchmark": {"concurrency_points": [1]}}),
                encoding="utf-8",
            )

            completed = self.invoke_report(run_dir)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            markdown = (run_dir / "summary.md").read_text(encoding="utf-8")
            self.assertIn("Run status: **PASS**", markdown)
            self.assertIn("## Throughput scaling", markdown)

    def test_missing_configured_point_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "benchmark-c001.json").write_text(
                json.dumps(valid_row()), encoding="utf-8"
            )
            (run_dir / "experiment.json").write_text(
                json.dumps({
                    "benchmark": {
                        "concurrency_points": [1, 30],
                        "expected_tags": ["c001", "c030"],
                    }
                }),
                encoding="utf-8",
            )

            completed = self.invoke_report(run_dir)
            self.assertEqual(completed.returncode, 2)
            markdown = (run_dir / "summary.md").read_text(encoding="utf-8")
            self.assertIn("Missing expected concurrency point(s): 30", markdown)
            self.assertIn("Missing expected benchmark tag(s): `c030`", markdown)


class ExternalHarnessTests(unittest.TestCase):
    def test_worker_exceptions_become_failed_records(self) -> None:
        original = bench_external.do_request

        def explode(_base: str, _prompt: str, max_tokens: int = 256):
            del max_tokens
            raise RuntimeError("synthetic worker failure")

        bench_external.do_request = explode
        try:
            records = bench_external.run_phase(
                "http://127.0.0.1:1/", ["prompt"], concurrency=3, per_worker=4
            )
        finally:
            bench_external.do_request = original

        self.assertEqual(len(records), 12)
        self.assertTrue(all(record["ok"] is False for record in records))
        self.assertTrue(all("synthetic worker failure" in record["error"] for record in records))

    def test_success_requires_every_expected_request(self) -> None:
        self.assertTrue(bench_external.successful_run(20, 0, 20, True))
        self.assertTrue(bench_external.successful_run(20, 0, 20, None))
        self.assertFalse(bench_external.successful_run(0, 0, 20, None))
        self.assertFalse(bench_external.successful_run(19, 1, 20, True))
        self.assertFalse(bench_external.successful_run(20, 0, 20, False))

    def test_empty_prompt_corpus_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.jsonl"
            path.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "prompt corpus is empty"):
                bench_external.load_prompts(path)


class ProvenanceTests(unittest.TestCase):
    @staticmethod
    def load_capture(path: Path):
        spec = importlib.util.spec_from_file_location("capture_env_test", path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_non_git_directory_is_unknown_not_clean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scripts = root / "scripts"
            scripts.mkdir()
            script = scripts / "capture_env.py"
            script.write_text((SCRIPTS / "capture_env.py").read_text(encoding="utf-8"), encoding="utf-8")
            module = self.load_capture(script)
            self.assertEqual(module.git_provenance(root), (None, None))

    def test_clean_and_dirty_git_states_are_distinguished(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scripts = root / "scripts"
            scripts.mkdir()
            script = scripts / "capture_env.py"
            script.write_text((SCRIPTS / "capture_env.py").read_text(encoding="utf-8"), encoding="utf-8")
            (root / "tracked.txt").write_text("clean\n", encoding="utf-8")
            (root / ".gitignore").write_text("__pycache__/\n*.pyc\n", encoding="utf-8")
            self.assertEqual(run("git", "init", "-q", cwd=root).returncode, 0)
            self.assertEqual(run("git", "config", "user.name", "ehzawad", cwd=root).returncode, 0)
            self.assertEqual(run("git", "config", "user.email", "test@example.invalid", cwd=root).returncode, 0)
            self.assertEqual(run("git", "add", ".", cwd=root).returncode, 0)
            self.assertEqual(run("git", "commit", "-qm", "fixture", cwd=root).returncode, 0)

            module = self.load_capture(script)
            commit, dirty = module.git_provenance(root)
            self.assertIsNotNone(commit)
            self.assertFalse(dirty)

            (root / "tracked.txt").write_text("dirty\n", encoding="utf-8")
            commit_after, dirty_after = module.git_provenance(root)
            self.assertEqual(commit_after, commit)
            self.assertTrue(dirty_after)


class ShellOrchestratorTests(unittest.TestCase):
    def make_fake_tools(self, root: Path) -> tuple[Path, Path]:
        fake_bin = root / "fake-bin"
        fake_bin.mkdir()
        log = root / "calls.log"

        python = fake_bin / "python3"
        python.write_text(
            """#!/usr/bin/env bash
set -u
echo "$*" >> "$FAKE_CALL_LOG"
if [[ "$*" == *"scripts/benchmark.py"* && "$*" == *"--tag c030"* ]]; then
  exit 2
fi
if [[ "$*" == *"scripts/benchmark.py"* && "$*" == *"--tag c024"* ]]; then
  exit 2
fi
if [[ "$*" == *"scripts/bench_external.py"* && "$*" == *"-c016"* ]]; then
  exit 2
fi
exit 0
""",
            encoding="utf-8",
        )
        python.chmod(0o755)

        nvidia_smi = fake_bin / "nvidia-smi"
        nvidia_smi.write_text("#!/usr/bin/env bash\necho 0\n", encoding="utf-8")
        nvidia_smi.chmod(0o755)

        for name, body in {
            "curl": "#!/usr/bin/env bash\nexit 0\n",
            "fuser": "#!/usr/bin/env bash\nexit 0\n",
            "sleep": "#!/usr/bin/env bash\nexit 0\n",
            "setsid": (
                "#!/usr/bin/env bash\n"
                "echo \"setsid $*\" >> \"$FAKE_CALL_LOG\"\n"
                "exec \"$REAL_SETSID\" \"$REAL_SLEEP\" 300\n"
            ),
        }.items():
            tool = fake_bin / name
            tool.write_text(body, encoding="utf-8")
            tool.chmod(0o755)

        return fake_bin, log

    def fake_env(self, fake_bin: Path, log: Path) -> dict[str, str]:
        env = dict(os.environ)
        env["PATH"] = f"{fake_bin}:{env['PATH']}"
        env["FAKE_CALL_LOG"] = str(log)
        real_setsid = shutil.which("setsid")
        real_sleep = shutil.which("sleep")
        if not real_setsid or not real_sleep:
            raise RuntimeError("tests require setsid and sleep")
        env["REAL_SETSID"] = real_setsid
        env["REAL_SLEEP"] = real_sleep
        return env

    def test_top_level_benchmark_continues_but_returns_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_bin, log = self.make_fake_tools(root)
            output = root / "run"
            completed = run(
                "bash",
                "run.sh",
                "benchmark",
                str(output),
                cwd=REPO,
                env=self.fake_env(fake_bin, log),
            )
            self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
            calls = log.read_text(encoding="utf-8")
            self.assertIn("--tag c001", calls)
            self.assertIn("--tag c030", calls)
            self.assertIn("--tag c100", calls)
            self.assertIn("scripts/report.py", calls)

    def test_concurrency_sweep_returns_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_bin, log = self.make_fake_tools(root)
            output = root / "sweep"
            completed = run(
                "bash",
                "scripts/sweep_concurrency.sh",
                str(output),
                cwd=REPO,
                env=self.fake_env(fake_bin, log),
            )
            self.assertEqual(completed.returncode, 1, completed.stdout + completed.stderr)
            calls = log.read_text(encoding="utf-8")
            self.assertIn("--tag c024", calls)
            self.assertIn("--tag c128", calls)
            self.assertIn("--tag c032b", calls)
            self.assertIn("scripts/report.py", calls)

    def test_engine_fair_propagates_client_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_bin, log = self.make_fake_tools(root)
            output = root / "engine-fair"
            env = self.fake_env(fake_bin, log)
            env.update({"CLIENTS": "1 8 16", "MEASURED": "1", "WARMUP": "0"})
            completed = run(
                "bash",
                "scripts/engine_fair.sh",
                str(output),
                "llamacpp",
                cwd=REPO,
                env=env,
            )
            self.assertEqual(completed.returncode, 1, completed.stdout + completed.stderr)
            calls = log.read_text(encoding="utf-8")
            self.assertIn("--tag llamacpp-bf16-c001", calls)
            self.assertIn("--tag llamacpp-bf16-c008", calls)
            self.assertIn("--tag llamacpp-bf16-c016", calls)

    def test_model_scale_marks_partial_models_and_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_bin, log = self.make_fake_tools(root)
            output = root / "model-scale"
            env = self.fake_env(fake_bin, log)
            env.update({
                "CLIENTS": "1 16",
                "MEASURED": "1",
                "WARMUP": "0",
                "NP": "1",
                "SLOTCTX": "64",
            })
            completed = run(
                "bash",
                "scripts/model_serve_bench.sh",
                str(output),
                cwd=REPO,
                env=env,
            )
            self.assertEqual(completed.returncode, 1, completed.stdout + completed.stderr)
            calls = log.read_text(encoding="utf-8")
            for model in ("qwen3-4b", "qwen3.5-9b", "gemma-4-e2b"):
                self.assertIn(f"--tag {model}-c001", calls)
                self.assertIn(f"--tag {model}-c016", calls)
                status = json.loads(
                    (output / f"status-{model}.json").read_text(encoding="utf-8")
                )
                self.assertEqual(status["status"], "benchmark_failed")

    def test_engine_wrappers_use_scoped_cleanup_and_failure_exit(self) -> None:
        for name in ("engine_fair.sh", "engine_compare.sh", "model_serve_bench.sh"):
            with self.subTest(script=name):
                text = (SCRIPTS / name).read_text(encoding="utf-8")
                self.assertNotIn("pkill -f", text)
                self.assertIn('exit "$fail"', text)
                self.assertIn("if ! python3 scripts/bench_external.py", text)
                self.assertIn("stop_group", text)

    def test_shell_syntax(self) -> None:
        completed = run(
            "bash",
            "-n",
            "run.sh",
            "scripts/sweep_concurrency.sh",
            "scripts/engine_fair.sh",
            "scripts/engine_compare.sh",
            "scripts/model_serve_bench.sh",
            cwd=REPO,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
