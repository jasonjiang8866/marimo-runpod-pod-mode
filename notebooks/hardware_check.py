import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch
    return mo, torch


@app.cell
def _(mo, torch):
    mo.md(f"""
    # RunPod hardware check

    PyTorch **{torch.__version__}** · CUDA runtime **{torch.version.cuda}** ·
    Visible GPUs **{torch.cuda.device_count()}**

    The startup check runs a tiny forward/backward operation on every visible
    device. This notebook displays hardware without downloading a model or
    starting training.
    """)
    return


@app.cell
def _(mo, torch):
    mo.ui.table([
        {"device": _i, "name": torch.cuda.get_device_name(_i),
         "VRAM GiB": round(torch.cuda.get_device_properties(_i).total_memory / 1024**3, 2),
         "compute capability": str(torch.cuda.get_device_capability(_i))}
        for _i in range(torch.cuda.device_count())
    ])
    return


if __name__ == "__main__":
    app.run()
