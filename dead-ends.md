# Dead ends

Things we measured that did not help on this card and model. Save yourself the time.

## Configuration

| tried | result |
|---|---|
| Vulkan backend instead of SYCL | 208 vs 902 t/s prefill, 7.6 vs 35.8 t/s decode |
| Level Zero env tweaks (`*_USE_IMMEDIATE_COMMANDLISTS`, `*_BATCH_SIZE`) | 0.9% spread, byte-identical output; already the default |
| Newer upstream tags for speed (b10901, b10950, b11095) | performance-neutral; bump for correctness fixes |
| Longer MTP drafts on Tier A (n-max > 3) | slower: each verified token costs ~14 ms of gated-delta work |
| n-max above 7 on Tier B | +2-3%, inside noise |
| Q5_K_M instead of Q4_K_M | slower (30.2 vs 32.7 t/s) and no VRAM left for context |
| q8_0 target KV | not worth it; quantize only the draft KV |
| `LLAMA_KV_PAD` | no gain |
| B70 overclock | fused at 2800 MHz, not power- or thermal-limited |

## A correction

An early test said the oneAPI 2026 toolchain was ~14x slower (a 400-token request took 413 s vs 29 s). That
was an artifact of tag b10950. On b11095, 2026.1 is fine, and Tier B requires it.

## Kernels and drafting

| tried | result |
|---|---|
| Exact int8 XMX GEMM for prefill | 2x slower than the f16 path |
| oneDNN u4 (int4) matmul path | no gain at 6 columns or fewer |
| Fused multi-token gated-delta step | no gain |
| Lazy gated-delta rollback | no gain |
| Sliding-window attention for the draft | no gain |
| Retraining the MTP head | failed its quality gate twice |
| DFlash2 drafter trained on the base model | dead on the fine-tuned 709-L |
| Scales held in registers inside `joint_matrix_apply` | wrong results and not faster |
| `LLAMA_MTP_DEFER_CATCHUP` | broken; never enable |
