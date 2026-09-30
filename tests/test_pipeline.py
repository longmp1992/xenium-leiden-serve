import os
from pathlib import Path

import pandas as pd
import pytest

from xenium_leiden.pipeline import QCParams, run_pipeline
from xenium_leiden.spatial import plot_spatial_clusters, read_cells_parquet

H5 = os.environ.get("XENIUM_TEST_H5")


@pytest.mark.skipif(not H5 or not Path(H5).exists(), reason="set XENIUM_TEST_H5 to a cell_feature_matrix.h5")
def test_cluster_csv_format(tmp_path):
    res = run_pipeline(H5, tmp_path, QCParams(), sample_tag="t")
    df = pd.read_csv(res.cluster_csv)
    assert list(df.columns) == ["Barcode", "Cluster"]
    assert df["Barcode"].is_unique
    assert df["Cluster"].min() == 1
    assert df["Cluster"].nunique() == res.stats["n_clusters"]
    assert len(df) == res.stats["n_cells_qc"]


def test_spatial_plot(tmp_path):
    cells = pd.DataFrame({"cell_id": [b"a-1", b"b-1", b"c-1"], "x_centroid": [0.0, 10.0, 5.0],
                          "y_centroid": [0.0, 5.0, 10.0], "cell_area": [1.0, 2.0, 3.0]})
    cells.to_parquet(tmp_path / "cells.parquet")
    clusters = pd.DataFrame({"Barcode": ["a-1", "b-1"], "Cluster": [1, 2]})
    png, pdf, summary = plot_spatial_clusters(read_cells_parquet(tmp_path / "cells.parquet"),
                                              clusters, tmp_path, "t")
    assert png.exists() and pdf.exists()
    assert "matched 2" in summary
    with pytest.raises(ValueError, match="No cell_id"):
        plot_spatial_clusters(read_cells_parquet(tmp_path / "cells.parquet"),
                              pd.DataFrame({"Barcode": ["zzz"], "Cluster": [1]}), tmp_path, "t")
