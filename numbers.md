# Measured numbers

One Arc Pro B70 (32 GB), Qwen3.8-27B (Fable 709-L) Q4_K_M with MTP heads, BF16 mmproj, f16 target KV, 131k
context, `-np 1`. Decode numbers marked "real traffic" come from an agent running a fixed coding task against
the seat, with the conversation at 30-75k tokens of context.

## Tier A (upstream b11095 + 3 patches)

| workload | speed |
|---|---|
| Text prefill, fresh ~40k-token prompt | ~900-1000 t/s |
| Image prefill (1132x2015 photo, ~2.2k visual tokens) | ~370 t/s warm |
| Video prefill (~50k tokens of frames) | ~420 t/s |
| 4K image (8192-token cap), cold | ~146 t/s, then ~0.6 s on re-send |
| Decode, shallow context, MTP on | ~42 t/s |
| Decode, real traffic at ~40k depth, MTP on | ~36.5 t/s (87-89% acceptance) |
| Decode, MTP off | ~26 t/s |

## Tier B step by step

Each row is a control-vs-candidate A/B on the same workload, back to back, and was promoted to production
before the next row was tried. Decode is real traffic at 30-75k depth.

| step | control | candidate |
|---|---|---|
| XMX q4_K matmul for verify batches | 38.2 t/s | 41.1 t/s |
| + XMX q6_K matmul and XMX flash attention | 41.8 | 52.2 |
| MTP draft n-max 5 to 7 | 55.1 | 59.5 |
| + command-graph replay, fast top-k | 60.6 | 65.3 |
| + split-K small-row MMVQ, gated-delta fusions | 59.2 | 63.2 |
| + register-fed DPAS (`*_DIRECT`) | | ~69 |
| + cheaper draft tokens (q8_0 draft KV, XMX q8_0 FA, no logit pad) | 69.6 | 72.3 |
| later host-traffic, prompt-cache and fewer-launch commits | | **~75-80** |

Prefill on fresh prompts:

| step | 40k tokens | 64k tokens |
|---|---|---|
| before chunked gated-delta | 913 t/s | 684 t/s |
| + chunked (WY) gated-delta prefill | 986 | 759 |
| + mask-upload skip, direct uploads, fast q6_K dequant | 1,134 | |
| + causal FA without a mask (bit-identical) | 1,200 | |
| final build | **~1,230** | |

Where the time goes at the end: the decode step is ~98% GPU-bound, and the verify graph is ~80% of it. Host
overhead was not the lever at any point.

## What the three patches fix

**0001, MTP across images and video.** Upstream's MTP drafter skips embedding batches without updating its
stored hidden row, so the first text token after an image pairs with a stale hidden state and acceptance
collapses in any chat with media. The patch copies the target's last hidden row for media chunks. Upstream's
later #28715 fixes the drafter's position accounting; it is complementary, and b11095 has both.

**0002, prompt-cache switching.** Upstream only consults the `--cache-ram` cache when the slot would lose at
least half its context, so returning to a long image chat that shares a system prompt re-prefills everything
after the prefix (2-3 minutes with images). The patch switches to a cached state whenever it covers more of
the prompt than the current slot. Re-sending a 4K image conversation went from ~49 s of prefill to ~0.6 s.

**0003, the mrope position warning.** mrope gives every row of an image the same temporal position, so the
recurrent memory's "one position per token" check fires twice on every image and buries the log. The patch
warns only on real backtracks for mrope models and leaves the check unchanged for other recurrent models. It
silences the noise; it does not fix the separate `seq_rm` rollback bug tracked upstream in #28007.
