"""Spatial plot of Leiden clusters using Xenium cells.parquet centroids."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402

CELL_ID_COL = "cell_id"
X_COL = "x_centroid"
Y_COL = "y_centroid"


def read_cells_parquet(path: str | Path) -> pd.DataFrame:
    """Read cell_id / x_centroid / y_centroid from a Xenium cells.parquet."""
    path = Path(path)
    try:
        import pyarrow.parquet as pq

        names = set(pq.read_schema(path).names)
    except Exception as exc:  # noqa: BLE001 - surface a readable message to the user
        raise ValueError(f"Could not read {path.name} as a parquet file: {exc}") from exc
    missing = {CELL_ID_COL, X_COL, Y_COL} - names
    if missing:
        raise ValueError(
            f"{path.name} is missing column(s) {sorted(missing)}; expected a Xenium cells.parquet "
            f"with {CELL_ID_COL}, {X_COL}, {Y_COL}."
        )
    df = pd.read_parquet(path, columns=[CELL_ID_COL, X_COL, Y_COL])
    ids = df[CELL_ID_COL]
    if ids.map(lambda v: isinstance(v, bytes)).any():
        ids = ids.map(lambda v: v.decode() if isinstance(v, bytes) else v)
    df[CELL_ID_COL] = ids.astype(str)
    return df


def cluster_palette(n: int) -> list[str]:
    """Scanpy-style categorical palette: tab20 up to 20 clusters, godsnot_102 beyond."""
    if n <= 20:
        cmap = plt.get_cmap("tab20" if n > 10 else "tab10")
        return [to_hex(cmap(i)) for i in range(n)]
    from scanpy.plotting.palettes import godsnot_102

    return [godsnot_102[i % len(godsnot_102)] for i in range(n)]


def plot_spatial_clusters(
    cells: pd.DataFrame,
    clusters: pd.DataFrame,
    out_dir: str | Path,
    tag: str,
    point_size: float = 1.0,
    show_unassigned: bool = True,
) -> tuple[Path, Path, str]:
    """Scatter cell centroids coloured by cluster. Returns (png, pdf, match summary)."""
    out_dir = Path(out_dir)
    clusters = clusters.assign(Barcode=clusters["Barcode"].astype(str))
    merged = cells.merge(clusters, left_on=CELL_ID_COL, right_on="Barcode", how="left")
    assigned = merged["Cluster"].notna()
    n_matched = int(assigned.sum())
    if n_matched == 0:
        raise ValueError(
            "No cell_id in cells.parquet matches a Barcode in cluster.csv; "
            "make sure both files come from the same Xenium sample."
        )

    cats = sorted(merged.loc[assigned, "Cluster"].astype(int).unique())
    colors = dict(zip(cats, cluster_palette(len(cats))))

    x, y = merged[X_COL].to_numpy(), merged[Y_COL].to_numpy()
    width = max(np.ptp(x), 1.0)
    height = max(np.ptp(y), 1.0)
    fig_w = 10.0
    fig_h = float(np.clip(fig_w * height / width, 4.0, 20.0))
    fig, ax = plt.subplots(figsize=(fig_w + 2.5, fig_h))

    if show_unassigned and (~assigned).any():
        ax.scatter(x[~assigned], y[~assigned], s=point_size, c="#d9d9d9", linewidths=0,
                   rasterized=True, label=f"QC-removed ({int((~assigned).sum())})")
    sub = merged.loc[assigned]
    ax.scatter(sub[X_COL], sub[Y_COL], s=point_size, c=sub["Cluster"].astype(int).map(colors),
               linewidths=0, rasterized=True)

    handles = [plt.Line2D([], [], marker="o", ls="", color=colors[c], markersize=6, label=str(c))
               for c in cats]
    if show_unassigned and (~assigned).any():
        handles.append(plt.Line2D([], [], marker="o", ls="", color="#d9d9d9", markersize=6,
                                  label="QC-removed"))
    ax.legend(handles=handles, title="Cluster", loc="upper left", bbox_to_anchor=(1.01, 1),
              frameon=False, ncol=1 if len(handles) <= 25 else 2, fontsize=8)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # image coordinates, same orientation as Xenium Explorer
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title(f"{tag}: Leiden clusters ({len(cats)})")
    fig.tight_layout()

    png = out_dir / f"{tag}_spatial_clusters.png"
    pdf = out_dir / f"{tag}_spatial_clusters.pdf"
    fig.savefig(png, dpi=200)
    fig.savefig(pdf, dpi=300)
    plt.close(fig)

    n_cells = len(merged)
    n_cluster_rows = len(clusters)
    summary = (
        f"cells.parquet: {n_cells} cells; cluster.csv: {n_cluster_rows} cells; "
        f"matched {n_matched} ({100 * n_matched / max(n_cluster_rows, 1):.1f}% of clustered cells)"
    )
    return png, pdf, summary
