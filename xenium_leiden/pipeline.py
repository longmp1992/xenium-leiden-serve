"""QC + Leiden clustering of a Xenium cell_feature_matrix.h5 into a HistoSeg cluster.csv.

Same recipe as the lab's run_leiden_generic.py:
Gene Expression features only -> filter_cells(min_counts, min_genes) -> filter_genes(min_cells)
-> normalize_total -> log1p -> PCA(50) -> neighbors(15) -> Leiden (igraph, resolution 1.0).
cluster.csv has columns Barcode,Cluster with 1-based cluster ids.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import scanpy as sc


@dataclass
class QCParams:
    min_counts: int = 10
    min_genes: int = 5
    min_cells: int = 5
    resolution: float = 1.0
    n_pcs: int = 50
    n_neighbors: int = 15
    seed: int = 0


@dataclass
class RunResult:
    cluster_csv: Path
    qc_summary: Path
    bundle_zip: Path
    summary_text: str
    cluster_sizes: pd.DataFrame
    stats: dict = field(default_factory=dict)


def _log(progress: Callable[[str], None] | None, msg: str) -> None:
    print(msg, flush=True)
    if progress is not None:
        progress(msg)


def run_pipeline(
    h5_path: str | Path,
    out_dir: str | Path,
    params: QCParams | None = None,
    sample_tag: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> RunResult:
    params = params or QCParams()
    h5_path = Path(h5_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = sample_tag or h5_path.name.removesuffix(".h5").removesuffix("_cell_feature_matrix")

    _log(progress, f"Reading {h5_path.name}")
    try:
        adata = sc.read_10x_h5(str(h5_path), gex_only=False)
    except Exception as exc:  # noqa: BLE001 - surface a readable message to the user
        raise ValueError(
            f"Could not read {h5_path.name} as a 10x/Xenium cell_feature_matrix.h5: {exc}"
        ) from exc
    adata.var_names_make_unique()
    n_cells_raw, n_features_raw = adata.shape

    if "feature_types" in adata.var:
        ft = adata.var["feature_types"].astype(str)
        adata = adata[:, (ft == "Gene Expression").to_numpy()].copy()
    n_genes_raw = adata.n_vars
    if n_genes_raw == 0:
        raise ValueError("No 'Gene Expression' features found in the file.")

    sc.pp.calculate_qc_metrics(adata, percent_top=None, inplace=True)
    sc.pp.filter_cells(adata, min_counts=params.min_counts)
    sc.pp.filter_cells(adata, min_genes=params.min_genes)
    sc.pp.filter_genes(adata, min_cells=params.min_cells)
    n_cells_qc, n_genes_qc = adata.shape
    _log(progress, f"QC: {n_cells_raw} -> {n_cells_qc} cells, {n_genes_raw} -> {n_genes_qc} genes")
    if n_cells_qc < 3 or n_genes_qc < 2:
        raise ValueError(
            f"Too few cells/genes left after QC ({n_cells_qc} cells, {n_genes_qc} genes); "
            "lower the thresholds."
        )

    _log(progress, "Normalizing, log1p, PCA")
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    n_pcs = int(min(params.n_pcs, n_cells_qc - 1, n_genes_qc - 1))
    sc.pp.pca(adata, n_comps=n_pcs, random_state=params.seed)

    _log(progress, "Building neighbor graph")
    sc.pp.neighbors(
        adata,
        n_neighbors=min(params.n_neighbors, n_cells_qc - 1),
        n_pcs=n_pcs,
        random_state=params.seed,
    )

    _log(progress, f"Leiden clustering (resolution {params.resolution})")
    sc.tl.leiden(
        adata,
        resolution=params.resolution,
        key_added="leiden",
        flavor="igraph",
        n_iterations=2,
        directed=False,
        random_state=params.seed,
    )

    clusters = adata.obs["leiden"].astype(int) + 1
    df = pd.DataFrame({"Barcode": adata.obs_names.astype(str), "Cluster": clusters.to_numpy()})
    cluster_csv = out_dir / f"{tag}_cluster.csv"
    df.to_csv(cluster_csv, index=False)

    sizes = df["Cluster"].value_counts().sort_index()
    cluster_sizes = pd.DataFrame(
        {"Cluster": sizes.index, "Cells": sizes.to_numpy(),
         "Fraction (%)": np.round(100 * sizes.to_numpy() / sizes.sum(), 2)}
    )
    n_clusters = int(sizes.size)

    summary_lines = [
        f"input file: {h5_path.name}",
        f"cells raw: {n_cells_raw}",
        f"features raw: {n_features_raw} (Gene Expression: {n_genes_raw})",
        f"cells after QC (min_counts>={params.min_counts}, min_genes>={params.min_genes}): {n_cells_qc}",
        f"genes kept (Gene Expression, min_cells>={params.min_cells}): {n_genes_qc}",
        f"median transcripts/cell after QC: {np.median(adata.obs['total_counts']):.1f}",
        f"median genes/cell after QC: {np.median(adata.obs['n_genes_by_counts']):.1f}",
        f"normalize_total + log1p, PCA {n_pcs} PCs, {params.n_neighbors} neighbors, seed {params.seed}",
        f"leiden resolution {params.resolution} -> {n_clusters} clusters",
        "",
        sizes.rename_axis("Cluster").rename("Cells").to_string(),
    ]
    summary_text = "\n".join(summary_lines)
    qc_summary = out_dir / f"{tag}_qc_summary.txt"
    qc_summary.write_text(summary_text + "\n", encoding="utf-8")

    bundle_zip = out_dir / f"{tag}_leiden_outputs.zip"
    with zipfile.ZipFile(bundle_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(cluster_csv, cluster_csv.name)
        zf.write(qc_summary, qc_summary.name)

    _log(progress, f"Done: {n_clusters} clusters, {n_cells_qc} cells")
    return RunResult(
        cluster_csv=cluster_csv,
        qc_summary=qc_summary,
        bundle_zip=bundle_zip,
        summary_text=summary_text,
        cluster_sizes=cluster_sizes,
        stats={"n_cells_raw": n_cells_raw, "n_cells_qc": n_cells_qc,
               "n_genes_qc": n_genes_qc, "n_clusters": n_clusters},
    )


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("h5")
    ap.add_argument("-o", "--out-dir", default=".")
    ap.add_argument("--tag")
    ap.add_argument("--min-counts", type=int, default=10)
    ap.add_argument("--min-genes", type=int, default=5)
    ap.add_argument("--min-cells", type=int, default=5)
    ap.add_argument("--resolution", type=float, default=1.0)
    a = ap.parse_args()
    res = run_pipeline(a.h5, a.out_dir,
                       QCParams(a.min_counts, a.min_genes, a.min_cells, a.resolution),
                       sample_tag=a.tag)
    print(res.summary_text)


if __name__ == "__main__":
    main()
