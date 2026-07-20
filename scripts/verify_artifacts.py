#!/usr/bin/env python3
"""Verify both GGUFs without loading model tensors (council merge-convert lens).

Usage: verify_artifacts.py <llama_dir> <bf16_gguf> <q6_gguf>
"""
import hashlib
import sys

llama_dir, bf16_path, q6_path = sys.argv[1:4]
sys.path.insert(0, f"{llama_dir}/gguf-py")

import gguf  # noqa: E402

EXPECTED_TEMPLATE_SHA = (
    "64f85b198065d0fba2a81f37e10ed68161ce2c19a754c7100e67e0ca2ee9c326"
)


def value(reader, key):
    field = reader.get_field(key)
    assert field is not None, key
    return field.contents()


def check_common(reader, label):
    assert value(reader, "general.architecture") == "qwen3"
    assert value(reader, "tokenizer.ggml.pre") == "qwen2"

    tokens = value(reader, "tokenizer.ggml.tokens")
    assert len(tokens) == 151936
    assert tokens[151643] == "<|endoftext|>"
    assert tokens[151644] == "<|im_start|>"
    assert tokens[151645] == "<|im_end|>"
    assert tokens[151669] == "[PAD151669]"
    assert tokens[151935] == "[PAD151935]"

    assert value(reader, "tokenizer.ggml.bos_token_id") == 151643
    assert value(reader, "tokenizer.ggml.padding_token_id") == 151643
    assert value(reader, "tokenizer.ggml.eos_token_id") == 151645

    template = value(reader, "tokenizer.chat_template")
    assert hashlib.sha256(template.encode()).hexdigest() == EXPECTED_TEMPLATE_SHA

    names = {t.name for t in reader.tensors}
    assert "token_embd.weight" in names
    assert "output.weight" not in names  # tied output head
    print(f"[verify] {label}: arch/tied-head/vocab/special-tokens/template OK")


bf16 = gguf.GGUFReader(bf16_path)
check_common(bf16, "bf16")
assert value(bf16, "general.file_type") == gguf.LlamaFileType.MOSTLY_BF16.value
assert {t.tensor_type.name for t in bf16.tensors} <= {"BF16", "F32"}

q6 = gguf.GGUFReader(q6_path)
check_common(q6, "Q6_K")
assert value(q6, "general.file_type") == gguf.LlamaFileType.MOSTLY_Q6_K.value
q6_types = {t.tensor_type.name for t in q6.tensors}
assert "Q6_K" in q6_types
print(f"[verify] Q6_K tensor types present: {sorted(q6_types)}")
print("[verify] GGUF architecture, tied head, vocabulary, special tokens, template, and types: OK")
