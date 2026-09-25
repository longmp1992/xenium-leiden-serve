import os
from pathlib import Path

import pandas as pd
import pytest

from xenium_leiden.pipeline import QCParams, run_pipeline

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
