"""Gradio app for SciLifeLab Serve: Xenium cell_feature_matrix.h5 -> QC -> Leiden -> cluster.csv (+ spatial plot)."""

from __future__ import annotations

import os
import re
import shutil
import time
import uuid
import zipfile
from pathlib import Path

import gradio as gr
import pandas as pd

from xenium_leiden.pipeline import QCParams, run_pipeline
from xenium_leiden.spatial import plot_spatial_clusters, read_cells_parquet

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


def _path(f) -> Path:
    return Path(f if isinstance(f, str) else f.name)


def _spatial(parquet_file, cluster_csv: Path, point_size, show_unassigned):
    """Draw the spatial cluster plot next to cluster.csv; returns (png, [png, pdf], summary)."""
    src = _path(parquet_file)
    if src.suffix.lower() != ".parquet":
        raise gr.Error("Expected a Xenium cells.parquet file.")
    tag = cluster_csv.name.removesuffix("_cluster.csv")
    try:
        cells = read_cells_parquet(src)
        png, pdf, summary = plot_spatial_clusters(
            cells, pd.read_csv(cluster_csv), cluster_csv.parent, tag,
            point_size=float(point_size), show_unassigned=bool(show_unassigned),
        )
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    bundle = cluster_csv.parent / f"{tag}_leiden_outputs.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in (cluster_csv, cluster_csv.parent / f"{tag}_qc_summary.txt", png, pdf):
            if f.exists():
                zf.write(f, f.name)
    return str(png), [str(png), str(pdf)], summary


def run(h5_file, parquet_file, min_counts, min_genes, min_cells, resolution, point_size,
        show_unassigned, progress=gr.Progress()):
    if h5_file is None:
        raise gr.Error("Please upload a Xenium cell_feature_matrix.h5 file.")
    src = _path(h5_file)
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

    plot_img, plot_files, spatial_msg = None, None, "Upload cells.parquet to draw a spatial plot."
    if parquet_file is not None:
        progress(0.99, desc="Drawing spatial plot")
        plot_img, plot_files, spatial_msg = _spatial(
            parquet_file, result.cluster_csv, point_size, show_unassigned)

    return (
        str(result.cluster_csv),
        str(result.bundle_zip),
        result.summary_text,
        result.cluster_sizes,
        plot_img,
        plot_files,
        spatial_msg,
        str(result.cluster_csv),
    )


def redraw(parquet_file, cluster_csv, point_size, show_unassigned):
    if not cluster_csv or not Path(cluster_csv).exists():
        raise gr.Error("Run QC + Leiden first (or the run has expired); then draw the spatial plot.")
    if parquet_file is None:
        raise gr.Error("Please upload the matching Xenium cells.parquet file.")
    cluster_csv = Path(cluster_csv)
    png, files, msg = _spatial(parquet_file, cluster_csv, point_size, show_unassigned)
    tag = cluster_csv.name.removesuffix("_cluster.csv")
    return png, files, msg, str(cluster_csv.parent / f"{tag}_leiden_outputs.zip")


with gr.Blocks(title="Xenium Leiden → HistoSeg cluster.csv") as demo:
    gr.Markdown(
        "# Xenium Leiden clustering → HistoSeg `cluster.csv`\n"
        "Upload a standard Xenium **`cell_feature_matrix.h5`**. The app keeps Gene Expression "
        "features, applies cell/gene QC, then runs `normalize_total` → `log1p` → PCA (50) → "
        "kNN (15) → **Leiden** and returns a `cluster.csv` (`Barcode,Cluster`, 1-based) that "
        "HistoSeg can use directly together with `cells.parquet`.\n\n"
        "Optionally upload the matching **`cells.parquet`** to get a spatial plot of the clusters "
        "(cell centroids coloured by cluster; cells removed by QC in grey). It can be uploaded "
        "before the run or afterwards — then press *Draw spatial plot*."
    )
    last_csv = gr.State(None)
    with gr.Row():
        with gr.Column(scale=1):
            h5_in = gr.File(label="cell_feature_matrix.h5", file_types=[".h5"], type="filepath")
            parquet_in = gr.File(label="cells.parquet (optional, for spatial plot)",
                                 file_types=[".parquet"], type="filepath")
            with gr.Accordion("QC and clustering parameters", open=True):
                min_counts = gr.Number(10, label="min_counts per cell (transcripts)", precision=0, minimum=0)
                min_genes = gr.Number(5, label="min_genes per cell", precision=0, minimum=0)
                min_cells = gr.Number(5, label="min_cells per gene", precision=0, minimum=0)
                resolution = gr.Number(1.0, label="Leiden resolution", minimum=0.01)
            with gr.Accordion("Spatial plot options", open=False):
                point_size = gr.Slider(0.05, 10, value=1.0, step=0.05, label="Point size")
                show_unassigned = gr.Checkbox(True, label="Show QC-removed cells in grey")
            run_btn = gr.Button("Run QC + Leiden", variant="primary")
            plot_btn = gr.Button("Draw spatial plot")
        with gr.Column(scale=1):
            csv_out = gr.File(label="cluster.csv (for HistoSeg)")
            zip_out = gr.File(label="All outputs (cluster.csv + QC summary + spatial plot)")
            summary_out = gr.Textbox(label="QC summary", lines=12)
            sizes_out = gr.Dataframe(label="Cluster sizes", interactive=False)
    with gr.Row():
        with gr.Column():
            spatial_msg = gr.Markdown()
            spatial_img = gr.Image(label="Spatial plot", type="filepath", interactive=False)
            spatial_files = gr.File(label="Spatial plot (PNG + PDF)", file_count="multiple")

    run_btn.click(
        run,
        inputs=[h5_in, parquet_in, min_counts, min_genes, min_cells, resolution, point_size,
                show_unassigned],
        outputs=[csv_out, zip_out, summary_out, sizes_out, spatial_img, spatial_files,
                 spatial_msg, last_csv],
        concurrency_limit=1,
    )
    plot_btn.click(
        redraw,
        inputs=[parquet_in, last_csv, point_size, show_unassigned],
        outputs=[spatial_img, spatial_files, spatial_msg, zip_out],
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
