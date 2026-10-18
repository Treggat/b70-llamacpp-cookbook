# B70 llama.cpp Cookbook

How to run a dense 27B hybrid (the Qwen3.8-27B family, any GGUF that ships the MTP tensors) on one **Intel Arc Pro B70 (32 GB)** with llama.cpp
SYCL, with fast prefill, image and video input, and MTP speculative decoding in the same `llama-server`.

This is a record of about a month of measured work on one box, written so you can skip the parts that did not
pay off. It has two tiers:

| tier | what you build | decode at 30-75k depth | fresh prefill, 40k prompt |
|---|---|---|---|
| **A. Stock + 3 patches** | upstream `b11095` + patches `0001`-`0003` from [`patches/`](patches/) | ~36-38 t/s | ~900-1000 t/s |
| **B. Full XMX build** | upstream `b11095` + all 27 patches in [`patches/`](patches/) | **~75-80 t/s** | **~1,230 t/s** |

Tier A is small, easy to rebase, and is what most people should run. Tier B roughly doubles decode, but it is
27 patches (~17.5k lines, mostly SYCL kernels) pinned to one upstream tag, most of it behind environment
switches, and it is **no longer maintained** (see [Status](#status)). Everything needed to build either tier is in
this repo.

## Start here

1. [Hardware checklist](recipes/01-hardware.md): ReBAR, PCIe link, VRAM budget.
2. [Build](recipes/02-build.md): Tier A and Tier B, both in Docker.
3. [Launch, Tier A](recipes/03-launch-stock.md): the server command, flag by flag.
4. [Launch, Tier B](recipes/04-launch-xmx.md): the XMX server command and its switches.
5. [Vision and video](recipes/05-vision-video.md): image budgets, sending video, frame control.
6. [Measuring honestly](recipes/06-measuring.md): how not to fool yourself.

Then [`numbers.md`](numbers.md) for the measured results and [`dead-ends.md`](dead-ends.md) for everything
that did not help.

## The short version

- **SYCL, not Vulkan.** Vulkan was 208 vs 902 t/s prefill and 7.6 vs 35.8 t/s decode on this card.
- **`-b 4096 -ub 2048`** is the biggest prefill lever. The default `-ub 512` leaves the card idle.
- **MTP (`--spec-type draft-mtp`)** takes decode from ~26 to ~42 t/s on Tier A. The MTP heads ship inside the
  GGUF; there is no separate draft model.
- **Stock llama.cpp cannot run MTP and vision together correctly.** Patch 0001 fixes that. Patch 0002 cuts
  the cost of returning to a long image chat from minutes to 0.6 s.
- **Q4_K_M is the right quant.** Q5 was slower and left no room for context.
- **Decode is about 98% GPU-bound.** Host-side tuning was never the lever; the kernels were.
- **XMX (`joint_matrix`) needs the oneAPI 2026.1 runtime.** 2025.3 reports no matrix aspect on Battlemage.

## Status

Finished, October 2026. We moved our main server to vLLM ([B70-Vllm-Cookbook](https://github.com/Treggat/B70-Vllm-Cookbook)),
so neither tier gets new work. Upstream has moved on;
if you port the patches to a newer tag, expect conflicts in `ggml-sycl` and `tools/server`.

## Credits

- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** (ggml-org and its contributors): everything here is a patch on top of it.
- **[praneshgo](https://github.com/praneshgo)**: patch `0004` is a cherry-pick of their probabilistic drafter,
  upstream PR [#27694](https://github.com/ggml-org/llama.cpp/pull/27694).
- **Anant Shrivastava**: patch `0005` is a cherry-pick of their SYCL MMVQ fusion work, upstream PR
  [#28931](https://github.com/ggml-org/llama.cpp/pull/28931).
- **[DavidAU](https://huggingface.co/DavidAU)** for the
  [Qwen3.8-27B 709-L merge](https://huggingface.co/DavidAU/Qwen3.8-27B-TWIN-TURBO-Fable-Cold-Fusion-709-L-Uncensored-NM-DAU-NEO-MTP-GGUF)
  the numbers were measured on.
- **Claude** (Anthropic; Claude Opus 5.5 and Claude Fable 5.1, through Claude Code): co-wrote the patches and kernels,
  ran the A/B measurements and helped write this cookbook. Each patch says which model assisted in its
  `Assisted-by:` line.

## Licence

The patches are derived from [llama.cpp](https://github.com/ggml-org/llama.cpp) and carry its MIT
licence. The text of this cookbook is MIT as well.
