# marimo on RunPod Pods

A configurable, authenticated marimo notebook environment for CPU Pods, consumer GPUs, workstation/MIG slices and datacenter GPUs. Python, marimo and PyTorch live in `/opt/venv`, so mounting storage at `/workspace` cannot hide them. Startup verifies a tiny forward/backward operation on **each visible device** before opening the editor.

This is a single-user notebook runtime, not a GPU scheduler. No RunPod API key is needed by the container. It does not create Pods, select live inventory, or stop billing automatically.

## Choose the runtime, then the hardware

Profiles are defined in [`profiles.json`](profiles.json). All builds target **Linux x86-64** (not ARM/GB200 hosts). marimo **0.25.2**, Python **3.12.14** and uv **0.13.7** are used across profiles.

| Profile / image suffix | PyTorch | CUDA wheel | Intended use |
|---|---|---|---|
| `cpu` | 2.15.0 | CPU | Editing, data preparation, small CPU experiments; no NVIDIA runtime needed |
| `cu126` | 2.14.1 | 12.6 | Older-driver hosts and older GPU architectures; also Ampere/Ada/Hopper when compatible |
| `cu128` **default** | 2.11.0 | 12.8 | Broad CUDA 12 compatibility, including RTX 5090 / RTX PRO Blackwell on suitable drivers |
| `cu132` | 2.15.0 | 13.2 | Current PyTorch on modern hosts, including Blackwell; requires a compatible CUDA 13 driver |

The older PyTorch pins are intentional compatibility options. PyTorch 2.15 no longer publishes CUDA 12.6 wheels; CUDA 12.8 is available in the 2.11 release series. A newer GPU does **not** imply its rented host has the newest driver. NVIDIA minor-version compatibility can allow newer CUDA runtimes within a major family, but PTX/JIT and new features may need newer drivers. Do not equate the `nvidia-smi` CUDA label with an installed toolkit, or reject every host whose label is below the wheel's minor version. Confirm with the actual workload after startup.

| Hardware tier examples (availability varies) | Profile starting point | Sizing guidance |
|---|---|---|
| CPU | `cpu` | Start with 4 vCPU / 8–16 GB RAM for editing and preprocessing |
| RTX 3090 / 4090, A5000 (24 GB) | `cu128`; `cu126` for older drivers | Small-model work; choose batch, sequence length and quantization for VRAM |
| RTX 5090 (32 GB), RTX PRO 6000 Blackwell MIG slice | `cu128` or `cu132` | Use **slice VRAM**, not the parent GPU's memory; a 24 GB slice is not a 96 GB device |
| A6000 / L40S (48 GB), A100 / H100 / H200 | `cu128` or `cu132` | Larger models; ensure CPU RAM and scratch storage also fit the workload |
| Multiple visible GPUs | Same driver-compatible profile | Startup checks each independently; DDP/NCCL and memory pooling are **not** implied |

These are selection guides, not measured training-capacity guarantees. The image does not change batch sizes automatically. Optional `MIN_GPU_MEMORY_GB` gates the visible memory of **each** GPU (GiB); leave headroom because nominal "24 GB" devices can report less usable memory.

## Build and deploy

No new registry image or RunPod template is claimed to exist until you publish one.

```bash
# On a Docker + Buildx machine (including an Apple Silicon client):
docker buildx bake cu128 --load
# Or choose cpu, cu126, cu132; build all profiles with `docker buildx bake`.

# Choose your own registry, then push:
IMAGE=ghcr.io/YOUR_USER/marimo-runpod-pod-mode docker buildx bake cu128 --push
```

A plain `docker build -t marimo-runpod:cu128 .` defaults to `cu128` on the build host's architecture; use `--platform linux/amd64` when appropriate. Build-time `RUNTIME_PROFILE` selects the wheels. Changing it in the Pod environment **does not** change the installed PyTorch.

In RunPod **Pod Templates**, configure:

1. **Image:** your published image, preferably `registry/repository@sha256:…`.
2. **Container start command:** leave empty to use the image's `CMD`. No `/start.sh` from the old RunPod base is required.
3. **HTTP port:** `8080`. If you change `MARIMO_PORT`, change the exposed port too.
4. **Storage mount:** `/workspace`. Allocate container disk for the pulled image and temporary files; 30 GB is a starting estimate, not a guarantee. Size persistent storage for your actual models and checkpoints.
5. **Authentication:** set `MARIMO_TOKEN_PASSWORD` using a RunPod secret reference, or mount a token file and set `MARIMO_TOKEN_PASSWORD_FILE`. With neither, marimo generates a random token shown in the Pod startup logs. Treat those logs as credentials.
6. Select CPU/GPU, GPU count, driver-compatible runtime and storage/datacenter options, then deploy. Open the HTTP connection and authenticate.

GPU images fail startup if CUDA cannot execute. `REQUIRE_GPU=0` is an explicit override for CPU-only CI or debugging; it is not needed for the `cpu` profile. A successful Docker build or HTTP health check does not validate real GPU performance.

### Local CPU smoke

```bash
docker buildx bake cpu --load
mkdir -p host_ws
docker run --rm -p 127.0.0.1:8080:8080 \
  -v "$PWD/host_ws:/workspace" -e SEED_EXAMPLES=1 marimo-runpod:cpu
```

Use the generated token from container logs. For local GPU use, select a GPU image and add `--gpus all` with the NVIDIA Container Toolkit installed on the host.

## Runtime settings

| Variable | Default | Behavior |
|---|---|---|
| `MARIMO_MODE` | `edit` | `edit` opens a notebook directory; `run` serves one notebook as an app |
| `NOTEBOOK_ROOT` | `/workspace` | Absolute, writable working directory |
| `NOTEBOOK_FILE` | unset | Existing absolute `.py` path; required in `run` mode |
| `MARIMO_PORT` | `8080` | TCP port, 1–65535 |
| `MARIMO_HOST` | `0.0.0.0` | Listen interface |
| `MARIMO_TOKEN_PASSWORD` | unset | Fixed token; passed to marimo through stdin, not argv |
| `MARIMO_TOKEN_PASSWORD_FILE` | unset | Readable nonempty token file; mutually exclusive with the variable above |
| `PUBLIC_HOST` | inferred on RunPod | `hostname:port` override for `--proxy`, e.g. `my.example:443`; no URL scheme |
| `MARIMO_BASE_URL` | empty | Optional reverse-proxy path prefix, e.g. `/notebooks` |
| `ALLOW_ORIGINS` | unset | Optional comma-separated origins, converted to repeated CLI flags; no wildcard default |
| `CACHE_ROOT` | `/workspace/.cache` | Parent for default Hugging Face, torch and uv caches |
| `HF_HOME`, `TORCH_HOME`, `UV_CACHE_DIR` | beneath `CACHE_ROOT` | Independently overridable cache paths |
| `REQUIRE_GPU` | `1` on GPU images, `0` on CPU | Fail if CUDA is unavailable; accepts `0` or `1` |
| `MIN_GPU_MEMORY_GB` | `0` | Minimum visible VRAM in GiB per GPU; finite and nonnegative |
| `SEED_EXAMPLES` | `0` | `1` copies the safe hardware notebook into `NOTEBOOK_ROOT/examples` if absent |

The proxy defaults to `${RUNPOD_POD_ID}-${MARIMO_PORT}.proxy.runpod.net:443`. marimo's normal origin checks remain in place; add only actual trusted origins if your proxy requires them. `/health` works without authentication, so there is no need to expose an unauthenticated editor for health checks.

Both modes use the baked Python environment (`--no-sandbox`). `run` mode is a read-only **UI**, not a security sandbox: notebook code still executes with the container's permissions. Treat the editor and its mounted data as a single trust boundary. The image runs as root for compatibility with RunPod-mounted storage; it is not a multi-tenant service.

## Storage and moving between hardware

| Storage | Recommended contents | Lifecycle / placement |
|---|---|---|
| Container filesystem (`/opt/venv`, `/opt/marimo`) | Baked packages and shipped examples | Recreated with the image; interactive package installs are ephemeral |
| Pod volume at `/workspace` | Notebooks, caches, local checkpoints | Retained across stop/start, deleted when the Pod is deleted |
| Network volume at `/workspace` | Persistent notebooks/checkpoints | Independent of Pod deletion; location constraints can limit GPU choice |
| Global Volume | Shared model weights and portable artifacts | Independent storage; may mount at `/workspace` or `/workspace-global` when combined with a network volume; verify the actual mount |

For Global Volume workflows, keep frequent small writes and compilation caches on suitable local/network storage. Set `NOTEBOOK_ROOT`, `CACHE_ROOT` and `HF_HOME` explicitly to match mounted paths and access permissions. Read-only model storage should be referenced by model path, with writable caches elsewhere. Do not put a shared virtualenv on a volume and reuse it blindly across CUDA profiles. Separate workload environments by image/runtime revision. The template never deletes or recreates volumes.

Data durability still requires backups. Terminating a Pod does not stop independent storage charges.

## Packages and example notebooks

The supported image contains **marimo, NumPy, PyTorch and their dependencies**. It does not claim to include Transformers, PEFT, TRL, bitsandbytes, vLLM or FlashAttention. This keeps hardware selection independent of workload-specific version constraints. CUDA runtime wheels are included; CUDA development headers, `nvcc` and C++ build tools are not. Workloads compiling extensions may need a derived development image with a matching toolkit.

`notebooks/hardware_check.py` is the supported, non-training example. The original `train_grpo.py` and `lora without regret.py` are retained as **historical experiments**, not validated against current TRL. They require additional libraries, download datasets/models and can start training when executed. Their hardcoded batch/generation settings are not suitable defaults for every hardware tier. They are not automatically copied into a fresh workspace.

For your own workload, build a derived image with a tested, pinned requirements file. Preserve the image's exact torch build while resolving additional dependencies:

```dockerfile
FROM your-registry/marimo-runpod@sha256:YOUR_PUBLISHED_DIGEST
COPY workload-requirements.txt /tmp/workload-requirements.txt
RUN python -c "import torch; print('torch==' + torch.__version__)" > /tmp/torch-constraint.txt \
 && uv pip install -r /tmp/workload-requirements.txt -c /tmp/torch-constraint.txt \
 && uv pip check
```

If a workload needs a different torch version or CUDA toolkit, create and validate a separate profile rather than upgrading torch in a running notebook. Add a workload smoke test before publishing that image.

## Validation and publication

- CI tests the real pinned marimo CLI: authentication, token handling, custom proxy/base path, app mode, validation errors, workspace preservation and CPU forward/backward.
- CI builds **all four** images, imports torch, checks server health/auth behind an empty `/workspace` bind mount, restarts, and checks shutdown. GPU images use `REQUIRE_GPU=0` **only for these CPU-hosted checks**.
- The manual **Publish selected runtime image** workflow builds, boots and tests one profile on the default branch, then pushes a source-SHA tag to GHCR and records its digest. Ensure the GHCR package is public or configure RunPod registry credentials.
- The editor dependencies are pinned in `requirements.txt` from `requirements.in`; torch/CUDA selections are pinned in `profiles.json`. The complete installed package inventory is recorded at `/opt/marimo/build-packages.txt`. Torch transitive dependencies and base-image tags are not fully locked: use the published **image digest** to reproduce a deployment.

```bash
uv venv .venv
uv pip install -r requirements.txt
.venv/bin/python scripts/install_runtime.py cpu
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/marimo check notebooks/hardware_check.py
# Docker-host check, after building:
bash scripts/container_smoke.sh marimo-runpod:cpu
```

Before accepting a new hardware tier, record the image digest, GPU/MIG name, visible VRAM, host driver, startup smoke output, notebook GPU execution, HTTP/WebSocket access through RunPod's real proxy and stop/start persistence. Multi-GPU workloads additionally need their own NCCL/DDP test. See the [review record](docs/review-2026-11-09.md) for findings and verification limits.

## Troubleshooting

- **CUDA unavailable / kernel image error:** inspect `nvidia-smi`, the selected profile and visible devices; use a compatible wheel/driver combination. Do not hide a GPU failure by silently using CPU.
- **Notebook cannot import torch:** confirm `sys.executable` is `/opt/venv/bin/python`; avoid creating an isolated sandbox without the same dependencies.
- **Proxy redirects or WebSocket failures:** check the exposed HTTP port, `PUBLIC_HOST`, `MARIMO_BASE_URL` and browser origin. RunPod proxy behavior needs a real-Pod check.
- **Out of memory:** reduce workload batch size, context/generation length or model size, or choose a larger slice/device. Startup smoke is deliberately tiny.
- **Packages disappear on redeploy:** put dependencies in a derived image; keep notebooks/checkpoints on persistent storage.

## Sources and license

Reviewed against upstream releases and documentation on **2026-11-09 (Singapore)**:

- [marimo 0.25.2](https://github.com/marimo-team/marimo/releases/tag/0.25.2), [CLI](https://docs.marimo.io/cli/), [authentication](https://docs.marimo.io/guides/deploying/authentication/)
- [PyTorch 2.15.0](https://github.com/pytorch/pytorch/releases/tag/v2.15.0), [wheel matrix](https://pytorch.org/get-started/previous-versions/), [release compatibility](https://github.com/pytorch/pytorch/blob/v2.15.0/RELEASE.md)
- [NVIDIA CUDA minor-version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
- [RunPod storage options](https://docs.runpod.io/pods/storage/types)

This repository is MIT licensed. Bundled software retains its upstream licenses, including NVIDIA CUDA components.
