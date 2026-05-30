# Childcare Market Segmentation

**Silhouette score 0.24 → 0.47** — UMAP cracked what KMeans and Ward's clustering couldn't. Unsupervised ML pipeline segmenting U.S. county-level childcare markets using socioeconomic and cost data from 2,500+ counties.

## Overview

Applied a full unsupervised ML pipeline to the National Database of Childcare Prices (TidyTuesday) to identify meaningful market segments across U.S. counties. Standard clustering methods (KMeans, Ward's hierarchical) hit a ceiling due to outliers and high-dimensional structure. Switching to UMAP for dimensionality reduction before clustering broke through — more than doubling the silhouette score without requiring outlier removal.

## Results

| Method | Silhouette Score | Notes |
|---|---|---|
| KMeans (k=4) | 0.24 | Outliers dominated cluster structure |
| Ward's Hierarchical | ~0.25 | Same outlier problem |
| **UMAP + KMeans** | **0.47** | Full dataset, no outlier removal |

## Approach

```
Raw Data (2,500+ U.S. counties)
    ↓
EDA + PCA          ← PCA revealed a socioeconomic gradient; no clean clusters
    ↓
KMeans / Ward's    ← Both converged on outlier-dominated clusters (silhouette ~0.24)
    ↓
UMAP               ← Nonlinear reduction exposed true latent structure
    ↓
KMeans on UMAP     ← Silhouette 0.47, interpretable segments
```

**4 segments identified:**
- High-cost urban markets (dense metro areas)
- Mid-cost suburban markets
- Rural low-cost markets
- Mixed-income transition counties

## Tech Stack

`Python` `scikit-learn` `UMAP-learn` `pandas` `matplotlib` `seaborn` `Google Colab`

## Notebooks

| Notebook | Description |
|---|---|
| `01_data_exploration.ipynb` | Initial data exploration and feature understanding |
| `02_eda_and_pca.ipynb` | EDA, PCA, socioeconomic gradient analysis |
| `03_clustering.ipynb` | KMeans, Ward's hierarchical, outlier analysis |
| `04_umap_segmentation.ipynb` | UMAP dimensionality reduction + final segmentation |

## Data

Dataset: [National Database of Childcare Prices](https://www.dol.gov/agencies/wb/topics/featured-childcare) via [TidyTuesday](https://github.com/rfordatascience/tidytuesday/tree/master/data/2023/2023-05-09). Public domain — included in `data/`.

## Team

Built as part of **BA820 (Unsupervised ML & Text Mining)** at Boston University with Hemanth Kumar Gopi, Shon Shaju, and Akhil Nair.

- M2 (EDA/PCA): Individual work — Kai Hung Tran
- M3 (Clustering pipeline): Collaborative
- M4 (UMAP segmentation): Individual work — Kai Hung Tran

Full team repo: [shonnvs-code/Team4-A1-BA820-Project](https://github.com/shonnvs-code/Team4-A1-BA820-Project/tree/main)
