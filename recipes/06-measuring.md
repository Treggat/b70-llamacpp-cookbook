# 6. Measuring honestly

- **Read the generated text, not just the t/s.** A fast kernel that produces garbage looks great in a
  benchmark. Every candidate we promoted was checked by reading its output and by a KL/perplexity comparison
  against the previous build on the same prompts.
- **Use the server's own timings** (`timings.prompt_per_second`, `timings.predicted_per_second`), not
  wall-clock time.
- **Burn one request after every load.** The first request pays the SYCL JIT cost.
- **Compare prefill only at the same cache warmth.** A warm prompt cache makes prefill look instant. Use
  fresh, uncached prompts (13.7k and 40k tokens were our standards) and say which you measured.
- **Measure decode at the depth you actually use.** Ours drops from ~42 t/s shallow to ~36 t/s at 40k on
  Tier A. All Tier B decode numbers here are at 30-75k depth.
- **Use real traffic for the final A/B.** Synthetic prompts gave different draft acceptance than an agent
  writing code. Our deciding test was a fixed coding task run by an agent against the server, control and
  candidate back to back, reporting decode t/s, ms per step, tokens per step and acceptance.
- **Variance is 5-13% run to run.** Treat differences under 5% as noise; repeat before you believe them.
- **Do not run `llama-bench` while the server can reload the model.** Two collisions caused GPU engine resets.
- **Switch sampler modes in your smoke test** (logprobs, `top_k 0`, tool calls). See
  [the Tier B rules](04-launch-xmx.md#rules-that-will-bite-you).
