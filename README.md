# Marimo on RunPod — GPU Notebook Template

> **One‑click, GPU‑ready Marimo IDE** on RunPod Pods. Ships with PyTorch + CUDA, Hugging Face stack, and a proxy‑aware startup so URLs work behind RunPod’s proxy.

---

## ✨ Highlights
- **GPU‑ready**: Based on `runpod/pytorch` (CUDA/torch preinstalled).
- **Proxy‑aware**: `start.sh` passes `--proxy <podId>-<port>.proxy.runpod.net:443` so redirects never leak internal IPs.
- **No Jupyter needed**: Launches Marimo headless as your browser IDE.
- **Fast, reproducible env**: Python virtualenv baked at **`/opt/venv`** (outside the volume) — no cold‑start re‑installs.
- **Persistent workspace**: Your notebooks live in `/workspace` (mounted volume), safe across restarts.

---

## 🚀 Quick start
1. **Deploy template** (RunPod Console → *Pods → Templates* → **Deploy**).  
   Or share a link: `https://runpod.io/console/deploy?template=<TEMPLATE_ID>`.
2. **Pick a GPU** (or CPU if you’re just editing code).  
3. **Expose port** `8080/http` and set a **Volume** (e.g., 10–50 GB) mounted at `/workspace`.
4. Click **Connect → HTTP**. You’ll land in Marimo instantly.

> **Tip:** If you want a fixed auth token for health checks/automation, set `MARIMO_TOKEN_PASSWORD` in the template’s **Environment Variables**.

---

## 🧱 Template defaults
- **Container image:** `docker.io/<you>/marimo-runpod:<tag>`
- **HTTP Ports:** `8080/http`
- **Volume:** `10–50 GB` at **`/workspace`** (persistent)
- **Container disk:** `20–30 GB` (ephemeral; must fit image + temp files)
- **GPU:** Optional (1x for acceleration)

---

## 🔧 Environment variables
| Variable | Default | Purpose |
|---|---:|---|
| `MARIMO_PORT` | `8080` | Marimo listen port |
| `NOTEBOOK_ROOT` | `/workspace` | Directory opened by Marimo |
| `MARIMO_TOKEN_PASSWORD` | *(unset)* | If set, enables token auth with this value; if unset, template runs with `--no-token` for simpler health checks |
| `ALLOW_ORIGINS` | `*` | CORS allowlist (comma‑separated) |
| `PUBLIC_HOST` | *(unset)* | Optional fallback public host if `RUNPOD_POD_ID` is not present |

> The startup script automatically builds the correct `--proxy` host on RunPod using `RUNPOD_POD_ID` and `MARIMO_PORT`.

---

## 🏗️ How it works
- **`/opt/venv`**: Python venv created at build time (not shadowed by the `/workspace` volume).
- **`start.sh`** (simplified):
  ```bash
  marimo edit \
    --headless \
    --host 0.0.0.0 \
    --port "$MARIMO_PORT" \
    --allow-origins "$ALLOW_ORIGINS" \
    ${MARIMO_TOKEN_PASSWORD:+--token-password "$MARIMO_TOKEN_PASSWORD"} \
    ${RUNPOD_POD_ID:+--proxy "${RUNPOD_POD_ID}-${MARIMO_PORT}.proxy.runpod.net:443"} \
    "$NOTEBOOK_ROOT"
  ```
- **Health**: With `--no-token` (default here), `/` returns `200` so RunPod marks the endpoint ready.

---

## 📦 Dockerfile sketch (for reference)
```dockerfile
FROM runpod/pytorch:1.0.0-cu1281-torch280-ubuntu2204
# Install uv (single binary)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /workspace
COPY requirements.txt .

# Create venv OUTSIDE the mount and install deps
RUN uv venv /opt/venv \
 && . /opt/venv/bin/activate \
 && uv pip install -r requirements.txt

ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH

# App
ENV MARIMO_PORT=8080
EXPOSE 8080
COPY start.sh /start.sh
RUN chmod +x /start.sh
CMD ["/start.sh"]
```

`requirements.txt` example:
```text
marimo[recommended]
torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128
transformers accelerate datasets
```

---

## 🔬 Sanity‑check the GPU (paste into a notebook cell)
```python
import torch, time
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")

# Tiny mixed‑precision training loop
Din,H,Dout,B=2048,4096,1024,1024
m=torch.nn.Sequential(
    torch.nn.Linear(Din,H), torch.nn.GELU(),
    torch.nn.Linear(H,Dout), torch.nn.GELU(),
    torch.nn.Linear(Dout,10)
).to("cuda" if torch.cuda.is_available() else "cpu")
opt=torch.optim.AdamW(m.parameters(),1e-3)
scaler=torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

device=next(m.parameters()).device
x=torch.randn(B,Din,device=device); y=torch.randint(0,10,(B,),device=device)
for _ in range(5):
    opt.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        loss=torch.nn.functional.cross_entropy(m(x),y)
    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
if device.type=="cuda": torch.cuda.synchronize()

n, t0 = 50, time.time()
for _ in range(n):
    opt.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        loss=torch.nn.functional.cross_entropy(m(x),y)
    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
if device.type=="cuda": torch.cuda.synchronize()
print({"loss": float(loss.item()), "steps": n, "samples/sec": round(n*B/(time.time()-t0),1)})
```

---

## 🧪 Local test before pushing
```bash
# build
docker build -t marimo-test:latest .
# run
docker run --rm -it -p 8080:8080 marimo-test:latest
# try with a host mount to simulate RunPod volume shadowing
mkdir -p host_ws
docker run --rm -it -p 8080:8080 -v "$PWD/host_ws:/workspace" marimo-test:latest
```

---

## 🩺 Troubleshooting
- **Endpoint not ready / redirects to 100.65.*:** Template already passes `--proxy`; ensure the port is exposed as **HTTP** and your browser URL is `https://<podid>-8080.proxy.runpod.net/`.
- **Venv missing at runtime:** Volume at `/workspace` can shadow image contents. We place the venv at `/opt/venv`; avoid recreating it on every boot.
- **Auth loops:** If you keep token auth on, add `?access_token=<MARIMO_TOKEN_PASSWORD>` to health checks. Or set `--no-token` (default here) for green health.
- **Container disk too small:** Increase container disk so the image + temp files fit (20–30 GB is safe for CUDA stacks).
- **Template not searchable:** Public templates often surface after some community runtime; share the direct deploy link meanwhile.

---

## 🔁 Updating the image in a Template
1. Push a new tag to your registry: `docker push <you>/marimo-runpod:v2`.
2. Edit the Pod Template → **Container image** → set to `...:v2` → **Save**.
3. Re‑deploy pods.

---

## 📄 License & attribution
This template bundles Marimo and PyTorch in a RunPod‑friendly container. Marimo is licensed under its upstream license. NVIDIA CUDA images are governed by NVIDIA’s license. Check the respective projects for terms.

---

**Enjoy!** If you want a Serverless variant (Hub repo) that runs a notebook and returns HTML/JSON, open an issue — it’s a small wrapper around `marimo run` with a `handler.py`. 
