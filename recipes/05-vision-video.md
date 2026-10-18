# 5. Vision and video

- **Use the BF16 mmproj.** bf16 to f16 clips the exponent range, a known source of NaNs and garbage on Qwen-VL
  vision encoders. In our A/B BF16 cost nothing (372 vs 371 t/s) and does not fall back to the CPU on SYCL.
- **`--image-min-tokens 1024 --image-max-tokens 8192`** lets full-resolution photos through: ~2.2k tokens for
  a phone photo, ~8.1k for 4K. Image tokens prefill about 2.4x slower per token than text; that is the price
  of the detail. With patch 0002, `--cache-ram` makes re-sends nearly free.
- **Send video as an `image_url` part** with `data:video/mp4;base64,...`. A `video_url` part returns HTTP 400.
- **The client must budget frames.** `llama-server` samples at `--video-fps` (default 4) and does not cap the frame
  count. Cost is roughly `frames / 2 x 1024` tokens.
- **Control frame count without restarting the server** by re-timing the clip:

  ```bash
  ffmpeg -i in.mp4 -vf "fps=S,setpts=(S/4)*PTS" -r 4 out.mp4   # S = targetFrames / duration
  ```

- **The model cannot hear.** Audio in the mp4 is discarded; transcribe it separately and send the text.
- **MTP and vision together need patch 0001.** Without it, draft acceptance collapses after the first image in
  a chat. With it we see 72-80% acceptance on image turns vs ~87% on text.
