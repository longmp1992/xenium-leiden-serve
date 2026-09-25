"""Gradio app for SciLifeLab Serve: Xenium cell_feature_matrix.h5 -> QC -> Leiden -> cluster.csv."""

from __future__ import annotations

import os
import re
import shutil
import time
import uuid
from pathlib import Path

import gradio as gr

from xenium_leiden.pipeline import QCParams, run_pipeline

PREFERRED_WORK_DIR = Path(os.environ.get("APP_DATA_DIR", "./project-vol")).resolve()
FALLBACK_WORK_DIR = Path("/tmp/project-vol")
MAX_FILE_SIZE = os.environ.get("MAX_UPLOAD_SIZE", "2gb")
RUN_RETENTION_HOURS = float(os.environ.get("RUN_RETENTION_HOURS", "24"))


def resolve_work_dir() -> Path:
    for candidate in (PREFERRED_WORK_DIR, FALLBACK_WORK_DIR):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_probe"
            probe.write_text("ok")
            probe.unlink()
            return candidate
        except OSError:
            continue
    raise RuntimeError("No writable work directory available")


WORK_DIR = resolve_work_dir()
RUNS_DIR = WORK_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)


def _cleanup_old_runs() -> None:
    cutoff = time.time() - RUN_RETENTION_HOURS * 3600
    for d in RUNS_DIR.iterdir():
        try:
            if d.is_dir() and d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass


def _sample_tag(filename: str) -> str:
    stem = Path(filename).name
    for suffix in (".h5", "_cell_feature_matrix"):
        stem = stem.removesuffix(suffix)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem) or "sample"


def run(h5_file, min_counts, min_genes, min_cells, resolution, progress=gr.Progress()):
    if h5_file is None:
        raise gr.Error("Please upload a Xenium cell_feature_matrix.h5 file.")
    src = Path(h5_file if isinstance(h5_file, str) else h5_file.name)
    if src.suffix.lower() != ".h5":
        raise gr.Error("Expected an .h5 file (Xenium cell_feature_matrix.h5).")

    _cleanup_old_runs()
    run_dir = RUNS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True)

    steps = iter(range(1, 100))
    log_lines: list[str] = []

    def report(msg: str) -> None:
        log_lines.append(msg)
        progress(min(next(steps) / 7, 0.99), desc=msg)

    params = QCParams(
        min_counts=int(min_counts),
        min_genes=int(min_genes),
        min_cells=int(min_cells),
        resolution=float(resolution),
    )
    try:
        result = run_pipeline(src, run_dir, params, sample_tag=_sample_tag(src.name), progress=report)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    return (
        str(result.cluster_csv),
        str(result.bundle_zip),
        result.summary_text,
        result.cluster_sizes,
    )


with gr.Blocks(title="Xenium Leiden → HistoSeg cluster.csv") as demo:
    gr.Markdown(
        "# Xenium Leiden clustering → HistoSeg `cluster.csv`\n"
        "Upload a standard Xenium **`cell_feature_matrix.h5`**. The app keeps Gene Expression "
        "features, applies cell/gene QC, then runs `normalize_total` → `log1p` → PCA (50) → "
        "kNN (15) → **Leiden** and returns a `cluster.csv` (`Barcode,Cluster`, 1-based) that "
        "HistoSeg can use directly together with `cells.parquet`."
    )
    with gr.Row():
        with gr.Column(scale=1):
            h5_in = gr.File(label="cell_feature_matrix.h5", file_types=[".h5"], type="filepath")
            with gr.Accordion("QC and clustering parameters", open=True):
                min_counts = gr.Number(10, label="min_counts per cell (transcripts)", precision=0, minimum=0)
                min_genes = gr.Number(5, label="min_genes per cell", precision=0, minimum=0)
                min_cells = gr.Number(5, label="min_cells per gene", precision=0, minimum=0)
                resolution = gr.Number(1.0, label="Leiden resolution", minimum=0.01)
            run_btn = gr.Button("Run QC + Leiden", variant="primary")
        with gr.Column(scale=1):
            csv_out = gr.File(label="cluster.csv (for HistoSeg)")
            zip_out = gr.File(label="All outputs (cluster.csv + QC summary)")
            summary_out = gr.Textbox(label="QC summary", lines=12)
            sizes_out = gr.Dataframe(label="Cluster sizes", interactive=False)

    run_btn.click(
        run,
        inputs=[h5_in, min_counts, min_genes, min_cells, resolution],
        outputs=[csv_out, zip_out, summary_out, sizes_out],
        concurrency_limit=1,
    )


def main() -> None:
    """Launch the Gradio app used by the SciLifeLab Serve deployment."""
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        show_api=False,
        max_file_size=MAX_FILE_SIZE,
        allowed_paths=[str(WORK_DIR)],
    )


if __name__ == "__main__":
    main()
