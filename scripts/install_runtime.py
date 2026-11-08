"""Install one explicit PyTorch wheel family; never choose by build-host GPU."""
import json
from pathlib import Path
import subprocess
import sys


def main():
    profiles = json.loads((Path(__file__).resolve().parents[1] / "profiles.json").read_text())
    name = sys.argv[1] if len(sys.argv) == 2 else ""
    if name not in profiles:
        raise SystemExit(f"Unknown runtime profile {name!r}; choose {', '.join(profiles)}")
    profile = profiles[name]
    subprocess.run([
        "uv", "pip", "install", "--no-cache",
        f"torch=={profile['torch']}", "--index-url",
        f"https://download.pytorch.org/whl/{profile['index']}",
    ], check=True)
    # Detect a mismatched CPU/CUDA wheel before publishing an image.
    import torch
    if torch.version.cuda != profile["cuda"]:
        raise SystemExit(f"Expected CUDA {profile['cuda']}, got {torch.version.cuda}")


if __name__ == "__main__":
    main()
