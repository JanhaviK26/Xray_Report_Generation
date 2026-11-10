# IU-Xray report generation

A frontal chest X-ray goes in. A findings paragraph comes out.

The encoder is a chest X-ray DenseNet-121 ([torchxrayvision](https://github.com/mlmed/torchxrayvision) `densenet121-res224-all`). It produces 49 spatial tokens, an MLP maps them into DistilGPT-2, and the decoder writes the findings section. A second model prepends the three most similar **training** reports before it writes. Both models were trained in `notebooks/cxr_report_gen_colab.ipynb`.

This is a research prototype on the Indiana University chest X-ray collection. It is not a medical device and must not be used for diagnosis.

## Try an image

Upload a frontal chest X-ray, or click one of the eight test images kept in `samples/`.

The page always returns the three closest training reports. 

## Test scores from the notebook

Held-out test split, 640 studies. Model rows are mean ± std over seeds 1, 2, and 3. Decoding for each seed was chosen on validation.

| System | CheXbert F1 | CheXbert F1, abnormal | BLEU-4 | ROUGE-L |
|---|---:|---:|---:|---:|
| Majority report | 0.000 | 0.000 | 0.060 | 0.252 |
| Nearest training image | 0.290 | 0.307 | 0.114 | 0.255 |
| No RAG | 0.318 ± 0.028 | 0.329 ± 0.024 | 0.117 ± 0.003 | 0.278 ± 0.002 |
| RAG-3 | 0.285 ± 0.034 | 0.298 ± 0.028 | 0.115 ± 0.006 | 0.277 ± 0.001 |

Full tables: `results/seed_summary_test.csv`, `results/seed_summary_val.csv`, `results/seed_runs_test.csv`, `results/seed_runs_val.csv`.

## What is in the repo

```
app/                  FastAPI page and the notebook's captioner
samples/              eight frontal test images, for trying the page
data/iu_xray/Kaggle/  studies.csv and feat_pooled.npy (retrieval index)
notebooks/            the Colab training notebook
results/              validation and test tables from that notebook
```

`studies.csv` row *i* is `feat_pooled.npy` row *i*. Retrieval uses only the training split. `feat_spatial.npy` is a local training cache and is not required to run the page.
