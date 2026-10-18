# 2. Build

Both tiers build in Intel's `deep-learning-essentials` image with the source mounted, using the same settings
as upstream's SYCL release job (fp16, `$ORIGIN` rpath, no native tuning).

## Tier A: upstream b11095 + three patches

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout b11095
git am /path/to/cookbook/patches/000[1-3]-*.patch

docker run --rm -it -v "$PWD:/src" -w /src \
  intel/deep-learning-essentials:2025.3.3-0-devel-ubuntu24.04 bash
```

Inside the container:

```bash
apt-get update -qq && apt-get install -y -qq git cmake ninja-build build-essential libcurl4-openssl-dev
cmake -B build-sycl -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_SYCL=ON -DGGML_SYCL_F16=ON \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_INSTALL_RPATH='$ORIGIN' -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
  -DGGML_NATIVE=OFF -DLLAMA_OPENSSL=OFF
cmake --build build-sycl --config Release
```

Ship `build-sycl/bin/*` and run it with the matching oneAPI runtime libraries on `LD_LIBRARY_PATH`.

## Tier B: the full XMX build

The XMX kernels use the SYCL matrix extension (`joint_matrix`), which needs the **oneAPI 2026.1** toolchain
and runtime. On 2025.3 the device does not report `aspect::ext_intel_matrix` for Battlemage and the XMX paths
stay off.

All 27 patches, in order, on the same tag. Patches `0001`-`0003` are Tier A; `0004` and `0005` are two upstream
PRs (#27694, #28931) that were not in `b11095` yet; the rest is ours.

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout b11095
git am /path/to/cookbook/patches/*.patch

docker run --rm -it -v "$PWD:/src" -w /src \
  intel/deep-learning-essentials:2026.1.3-devel-ubuntu24.04 bash
```

Inside the container:

```bash
apt-get update -qq
apt-get install -y -qq --no-install-recommends intel-oneapi-dnnl-devel-2026.0 \
  git libcurl4-openssl-dev cmake build-essential ninja-build ca-certificates
cmake -B build-sycl -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DGGML_SYCL_DNN=ON -DGGML_SYCL_XMX=ON \
  -DDNNL_DIR=/opt/intel/oneapi/dnnl/2026.0/lib/cmake/dnnl \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_INSTALL_RPATH='$ORIGIN' -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
  -DGGML_NATIVE=OFF -DLLAMA_OPENSSL=OFF
cmake --build build-sycl --config Release
```

`-DGGML_SYCL_XMX=ON` also builds the register-fed DPAS kernels as their own library
(`libggml-sycl-xmxd.so`); ship it next to `libggml-sycl.so`. Run Tier B with the oneAPI **2026.1** runtime
libraries on `LD_LIBRARY_PATH`.

## Tips

- **On Windows, clone with `git -c core.autocrlf=false clone ...`.** With CRLF conversion on, `git am` fails
  with "does not match index".
- Keep the build tree in a named Docker volume. A one-file change then rebuilds in about a minute.
- Exit code 0 does not prove a rebuild. Check the mtime of the object or `.so` you expected to change.
- Before deploying, diff the new bundle's file list against the build you are replacing, not against an old
  reference build. A missing `.so` shows up as a load failure only on the server.
