| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| qwen3 4B Q4_K - Medium         |   2.32 GiB |     4.02 B | CUDA       |  99 |   1 |           pp512 |     6005.02 ± 469.66 |
| qwen3 4B Q4_K - Medium         |   2.32 GiB |     4.02 B | CUDA       |  99 |   1 |           tg128 |        164.45 ± 0.13 |
| qwen3 4B Q5_K - Medium         |   2.69 GiB |     4.02 B | CUDA       |  99 |   1 |           pp512 |     5813.10 ± 394.86 |
| qwen3 4B Q5_K - Medium         |   2.69 GiB |     4.02 B | CUDA       |  99 |   1 |           tg128 |        152.77 ± 0.43 |
| qwen3 4B Q6_K                  |   3.07 GiB |     4.02 B | CUDA       |  99 |   1 |           pp512 |     5431.34 ± 330.46 |
| qwen3 4B Q6_K                  |   3.07 GiB |     4.02 B | CUDA       |  99 |   1 |           tg128 |        129.93 ± 0.10 |
| qwen3 4B Q8_0                  |   3.98 GiB |     4.02 B | CUDA       |  99 |   1 |           pp512 |     6368.28 ± 430.33 |
| qwen3 4B Q8_0                  |   3.98 GiB |     4.02 B | CUDA       |  99 |   1 |           tg128 |        124.15 ± 0.20 |
| qwen3 4B BF16                  |   7.49 GiB |     4.02 B | CUDA       |  99 |   1 |           pp512 |     6194.85 ± 658.59 |
| qwen3 4B BF16                  |   7.49 GiB |     4.02 B | CUDA       |  99 |   1 |           tg128 |         76.82 ± 0.07 |

build: 91d2fc3 (1)
