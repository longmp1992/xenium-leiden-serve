# Xenium Leiden → HistoSeg cluster.csv (SciLifeLab Serve)

Gradio app for SciLifeLab Serve. Upload a standard Xenium `cell_feature_matrix.h5`; the app

1. keeps `Gene Expression` features (drops negative-control probes / codewords),
2. QC: `filter_cells(min_counts=10)`, `filter_cells(min_genes=5)`, `filter_genes(min_cells=5)`,
3. `normalize_total` → `log1p` → PCA (50 PCs) → kNN graph (15 neighbors),
4. Leiden clustering (igraph, resolution **1.0**, seed 0),
5. returns `<sample>_cluster.csv` and `<sample>_qc_summary.txt` (+ a zip of both).

All thresholds and the resolution are editable in the UI.

`cluster.csv` format (what HistoSeg expects, same as the Xenium Explorer / GraphClust CSV):

```
Barcode,Cluster
aaabcnmo-1,1
aaabkidh-1,2
```

`Cluster` is 1-based. Cells removed by QC are not listed.

## Local run

```bash
pip install -r requirements.txt
python main.py                      # http://localhost:7860
python -m xenium_leiden.pipeline cell_feature_matrix.h5 -o out/   # CLI, same pipeline
```

Docker:

```bash
docker build -t xenium-leiden-serve .
docker run --rm -p 7860:7860 xenium-leiden-serve
```

## Deploy on SciLifeLab Serve

Pushing to `main` builds `ghcr.io/<owner>/xenium-leiden-serve:latest` (GitHub Actions).
On serve.scilifelab.se create an app of type **Gradio** (or custom Docker app) with that image,
port `7860`. Uploads are limited to `MAX_UPLOAD_SIZE` (default `2gb`); run folders under
`$APP_DATA_DIR/runs` are deleted after `RUN_RETENTION_HOURS` (default 24).

## Test

```bash
XENIUM_TEST_H5=/path/to/cell_feature_matrix.h5 pytest -q
```
