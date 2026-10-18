# 4. Launch, Tier B (the full XMX build)

This is the exact command our production server ran on the full Tier B build with the oneAPI 2026.1 runtime.
Most of what patches `0010`-`0027` add is **off by default** and turned on by an environment variable. With
no variables set the binary still differs from Tier A: the MMVQ fusions, GQA padding and Q6_K paths from
patches `0005`-`0007` and the bug fixes in later patches are always on, and patch `0004` adds the
`--spec-draft-sampling probabilistic` option.

```bash
export GGML_SYCL_ENABLE_GRAPH=1 GGML_SYCL_GRAPH_MAX_TOKENS=8 GGML_SYCL_GRAPH_ROT_ROWS=1 \
       GGML_SYCL_GRAPH_RECORD_AHEAD=1 GGML_SYCL_GRAPH_SIG_MEMO=1
export GGML_SYCL_XMX_Q4K=1 GGML_SYCL_XMX_Q6K=1 GGML_SYCL_XMX_Q80=1 \
       GGML_SYCL_XMX_Q4K_DIRECT=1 GGML_SYCL_XMX_Q6K_DIRECT=1 \
       GGML_SYCL_XMX_GATEUP=2 GGML_SYCL_XMX_GLUE=1 GGML_SYCL_XMX_KS_V2=1
export GGML_SYCL_XMX_FA=1 GGML_SYCL_XMX_FA_V2=1 GGML_SYCL_XMX_FA_Q8=1 GGML_SYCL_FA_CAUSAL=1
export GGML_SYCL_DQ_GEMM=1 GGML_SYCL_Q6K_DQ_FAST=1 GGML_SYCL_SMALLROW=1 \
       GGML_SYCL_GDN_CHUNKED=1 GGML_SYCL_GDN_FUSE=1 GGML_SYCL_GDN_GATHER=1 GGML_SYCL_CONV_STEP=1
export GGML_SYCL_ASYNC_INPUTS=2 GGML_SYCL_H2D_DIRECT=1 GGML_SYCL_TOPK_BATCH=1 GGML_SYCL_DNN_DETERMINISTIC=1
export LLAMA_KQ_MASK_HINT=3 LLAMA_KQ_MASK_HINT_NOFILL=1 LLAMA_KQ_MASK_FAST=1 \
       LLAMA_MTP_H_DEVICE=1 LLAMA_MTP_DRAFT_NOPAD=1 LLAMA_KV_ROT_DEVICE=1 LLAMA_TGT_TOPK_DEVICE=1 \
       LLAMA_SAMPLER_FAST_TOPK=1 LLAMA_BALLOC_FAST=1 LLAMA_GRAPH_REUSE_MULTI=1 \
       LLAMA_KV_SEQ_RM_FAST=1 LLAMA_SYNC_SKIP_IDLE=1
export LLAMA_PCACHE_POOL=1 LLAMA_PCACHE_CKPT_MOVE=1 LLAMA_CKPT_POOL=1 LLAMA_KV_CKPT_PARTIAL_SKIP=1

LD_LIBRARY_PATH=/path/to/llama-build:/path/to/oneapi-2026.1:$LD_LIBRARY_PATH \
llama-server \
  -m  Qwen3.8-27B-...-MTP-Q4_K_M-q8head-dh98k.gguf \
  --mmproj mmproj-BF16.gguf \
  -ngl 99 --load-mode dio -fa on -fit off \
  -np 1 --kv-unified -c 131072 \
  -b 4096 -ub 2048 \
  --cache-ram 16384 --ctx-checkpoints 8 \
  --image-min-tokens 1024 --image-max-tokens 8192 \
  --spec-type draft-mtp -ctkd q8_0 -ctvd q8_0 \
  --spec-draft-n-max 7 --spec-draft-p-min 0.5 --spec-draft-sampling probabilistic \
  --temp 1.0 --top-p 0.95 --top-k 20 \
  --jinja --reasoning-preserve
```

## What the switches do, by group

| group | switches | effect |
|---|---|---|
| Command-graph replay | `GGML_SYCL_ENABLE_GRAPH`, `GRAPH_*` | records the decode/verify graph once and replays it; ~91% of steps replay |
| XMX matmul | `XMX_Q4K`, `XMX_Q6K`, `XMX_Q80`, `*_DIRECT`, `GATEUP`, `GLUE`, `KS_V2` | `joint_matrix` and register-fed DPAS kernels for 2-8 column verify batches, the q8_0 LM head, split-K tables |
| XMX flash attention | `XMX_FA`, `XMX_FA_V2`, `XMX_FA_Q8`, `FA_CAUSAL` | decode FA on XMX, q8_0 draft KV, causal masking without uploading a mask (bit-identical) |
| Prefill | `DQ_GEMM`, `Q6K_DQ_FAST`, `SMALLROW`, `GDN_CHUNKED`, `GDN_FUSE`, `GDN_GATHER`, `CONV_STEP` | fused q4_K dequant + f16 GEMM for short ubatches, chunked (WY) gated-delta prefill, fusions |
| Host traffic | `ASYNC_INPUTS`, `H2D_DIRECT`, `KQ_MASK_*`, `*_DEVICE`, `SYNC_SKIP_IDLE` | keeps decode-step data on the device, skips mask uploads and idle syncs |
| Sampling | `TOPK_BATCH`, `TGT_TOPK_DEVICE`, `SAMPLER_FAST_TOPK` | top-k on the device |
| Prompt cache | `PCACHE_POOL`, `PCACHE_CKPT_MOVE`, `CKPT_POOL`, `KV_CKPT_PARTIAL_SKIP` | cheaper conversation switches: pooled save/restore instead of fresh zero-filled allocations |
| Determinism | `DNN_DETERMINISTIC` | forces oneDNN's deterministic GEMM (see below) |

Each switch went in after a same-workload A/B against the previous build; the gains are in
[`numbers.md`](../numbers.md#tier-b-step-by-step). The message at the top of each file in [`patches/`](../patches/)
describes what that patch adds and names its switches.

## The model file

`-q8head-dh98k` means the GGUF was rewritten with `scripts/b70/gguf_q8_head_draft_head.py`, which patch
[`0009`](../patches/0009-scripts-GGUF-tool-for-a-Q8_0-LM-head-plus-MTP-draft-.patch) adds to the tree:

```bash
python scripts/b70/gguf_q8_head_draft_head.py src.gguf dst.gguf 98304
```

It re-encodes `output.weight` from BF16 to Q8_0 and adds `nextn.draft_head`, the first 98,304 rows of that
head, which the MTP drafter uses instead of the full 248k vocab (the target still verifies over the full vocab;
98,304 rows cover ~99.95% of tokens). We also requantized the draft head and the MTP layer to Q4_K for another
-419 MiB; that step used a separate script and is optional.

## Rules that will bite you

- **`GGML_SYCL_GRAPH_MAX_TOKENS=8` is required with `XMX_Q4K_DIRECT`.** Recording 9-32 token ubatches makes
  the server exit.
- **`-ctkd q8_0 -ctvd q8_0` quantizes only the draft KV.** Target KV stays f16. Target q8_0 KV was a dead end.
- **`--spec-draft-n-max 7`, not 3.** The faster verify kernels moved the optimum. Above 7 the gain was 2-3%,
  inside noise.
- **oneDNN f16 GEMM is non-deterministic** for small M x N with large K (it picks a split-K kernel with float
  atomics). `GGML_SYCL_DNN_DETERMINISTIC=1` fixes it; without it, fixed-seed comparisons drift.
- **Any in-place weight repack must own every consumer of the tensor,** including multi-sequence ubatches and
  warmup.
- **Never enable `LLAMA_MTP_DEFER_CATCHUP`.** It is broken.
- **A 256-GRF XMX kernel next to 128-GRF kernels costs ~6.5 us per mode switch.** Keep register modes
  consistent inside a graph.
- **Smoke-test sampler changes,** not just output: request logprobs, `top_k 0` and tools between ordinary
  requests. A top-k re-attach crash (fixed in patch `0026`) passed fixed-seed screens, KL checks and real
  traffic before a logprobs request found it.
