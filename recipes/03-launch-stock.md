# 3. Launch, Tier A (stock + 3 patches)

```bash
LD_LIBRARY_PATH=/path/to/llama-build:/path/to/oneapi-2025.3:$LD_LIBRARY_PATH \
llama-server \
  -m  Qwen3.8-27B-...-MTP-Q4_K_M.gguf \
  --mmproj mmproj-BF16.gguf \
  -ngl 99 -fa on -fit off \
  -np 1 --kv-unified \
  -c 131072 \
  -b 4096 -ub 2048 \
  --cache-ram 16384 \
  --image-min-tokens 1024 --image-max-tokens 8192 \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.5 \
  --temp 1.0 --top-p 0.95 --top-k 20 \
  --jinja --reasoning-preserve
```

## Prefill

- **`-b 4096 -ub 2048`.** The biggest prefill lever: bigger matmuls per launch. It did not help every model
  equally (+22% on one model, +8% on another), and the compute buffer grows with `-ub`, so on a server already near the
  VRAM limit, requests failed with HTTP 502. Check your headroom.
- **`-fa on`.** Leave flash attention on.
- **`-np 1 --kv-unified`.** One slot, so the whole batch goes to one sequence and the KV cache is not split
  across idle slots.
- **f16 KV (the default).** We ran f16 at 131k context.

## Decode

- **MTP takes decode from ~26 to ~42 t/s** at shallow depth, with 87-89% draft acceptance on text.
- **`--spec-draft-n-max 3 --spec-draft-p-min 0.5`** is the tuned setting for this tier. We swept 8 draft
  geometries. The arch has 48 Gated-DeltaNet layers, so every extra verified token costs about 14 ms per step
  with the stock kernels; the limit is per-step cost, not the drafter. (Tier B's faster verify kernels move
  the best `n-max` to 7.)
- **Decode is bandwidth times quant size.** Without MTP every token reads all the weights, which puts a 27B at
  Q4_K_M near 40 t/s. IQ4_XS would project to ~47 and Q3_K_M to ~52, both with a quality cost.

## Prompt cache

`--cache-ram 16384` keeps recent conversation states in host RAM. With patch 0002 the server switches to a
cached state whenever it covers more of the new prompt than the current slot does, so going back to a long
conversation costs a restore instead of a re-prefill. See [`../patches/`](../patches/) and the patch notes in
[`numbers.md`](../numbers.md#what-the-three-patches-fix).

## Warm up

The first request after a load pays a one-time SYCL JIT cost (~30% slower on the first image). Send one
throwaway request before you measure anything.
