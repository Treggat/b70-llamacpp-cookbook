# 1. Hardware checklist

- **Resizable BAR on.** Check it in the BIOS and confirm with `lspci -vv` (a BAR the size of VRAM).
- **PCIe Gen4 x16, read in the right place.** The B70 has an onboard PCIe switch, so `lspci` on the GPU
  endpoint can report a misleading link. Read the link speed on the **upstream switch port** instead.
- **The link matters for loads and prompt-cache moves, not for decode.** We ran one card on a Gen3 x4 chipset
  slot (host-to-device ~2-3 GB/s) and one on Gen4 x16. Decode was the same; model loads and cache restores
  were slower on the narrow link.
- **Everything in VRAM.** 17.2 GiB Q4_K_M model + ~0.9 GiB BF16 mmproj + f16 KV at 131k context fits in 32 GB
  with `-ub 2048`. Any CPU offload (`-ngl` below the layer count) kills both prefill and decode.
- **No overclock.** The clock is fused at 2800 MHz and the card is not power- or thermal-limited under load.
  Leave it stock.
