"""Small forward/backward gate on every visible GPU, including MIG devices."""
import json
import math
import os
import platform
import sys


def check(torch, require_gpu, min_memory_gb=0):
    result = {
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "devices": [],
    }
    if not torch.cuda.is_available():
        if require_gpu:
            raise RuntimeError("CUDA is unavailable. Check GPU attachment, host driver and image profile; GPU images do not silently fall back to CPU.")
        devices = ["cpu"]
    else:
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    for device in devices:
        info = {"device": device}
        if device != "cpu":
            props = torch.cuda.get_device_properties(device)
            memory = props.total_memory / 1024**3
            info.update(name=props.name, memory_gib=round(memory, 2), capability=f"{props.major}.{props.minor}")
            if memory < min_memory_gb:
                raise RuntimeError(f"{device} has {memory:.2f} GiB, below MIN_GPU_MEMORY_GB={min_memory_gb}. MIG reports the visible slice, not the full GPU.")
        # Allocation alone cannot detect a missing kernel image or broken cuBLAS.
        x = torch.ones((32, 32), device=device, requires_grad=True)
        y = (x @ x).sum()
        y.backward()
        if y.item() != 32768 or not torch.all(x.grad == 64).item():
            raise RuntimeError(f"Numerical forward/backward check failed on {device}")
        if device != "cpu":
            torch.cuda.synchronize(device)
        info["smoke"] = "passed"
        result["devices"].append(info)
    return result


def main():
    try:
        required = os.environ.get("REQUIRE_GPU", "0" if os.environ.get("RUNTIME_PROFILE", "cpu") == "cpu" else "1")
        if required not in ("0", "1"):
            raise ValueError("REQUIRE_GPU must be 0 or 1")
        memory = float(os.environ.get("MIN_GPU_MEMORY_GB", "0"))
        if not math.isfinite(memory) or memory < 0:
            raise ValueError("MIN_GPU_MEMORY_GB must be finite and nonnegative")
        import torch
        print(json.dumps(check(torch, required == "1", memory)), flush=True)
    except Exception as exc:
        print(f"Hardware check failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
