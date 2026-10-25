# UA-FloodNet — Research notebooks

Companion research notebooks for **UA-FloodNet: Evaluating SAR–Optical Misregistration Robustness in Multitemporal Flood Mapping**.

> **Important:** These are archival, output-cleared Google Colab research notebooks, **not a verified end-to-end runnable release**. The uploads supplied for preparation did not include all referenced Python modules, data conversions, trained checkpoints, or downloaded GEOID-Flood imagery. Do not represent this package as independently reproducible until those dependencies are provided and the instructions tested in a clean Colab runtime.

## Contents and order

1. `notebooks/01_UA_FloodNet_Development.ipynb` — early environment, OMBRIA conversion, prototypes, V4 control/augmentation training, exploratory evaluation.
2. `notebooks/02_UA_FloodNet_Training_and_Figures.ipynb` — multi-seed training orchestration, qualitative outputs, architecture inspection, figure and table generation.
3. `notebooks/03_UA_FloodNet_Ablations_Benchmarks_External.ipynb` — architectural ablations, conventional U-Net comparisons, GEOID-Flood preprocessing and external inference, efficiency, statistics and plots.
4. `scripts/train_multiseed.py` — the standalone Python source from the second notebook's `%%writefile` cell, preserved without methodological changes. **It depends on other unavailable local modules.**
5. `AUDIT.md` — inspection findings and release limitations.

## Data and model dependencies

The notebooks expect Google Colab and specific Google Drive folders, archives, image arrays and trained weights. References include `ablation_model.py`, `data.py`, `train_v4.py`, and other scripts not supplied in these three notebooks. Paths under `/content/` or `/content/drive/MyDrive/UA_FloodNet` reflect the author's Colab setup; change these to match your environment. No dataset, weights or model outputs are bundled.

The article reports 694 OMBRIA 256×256 paired patches (499 train, 125 candidate validation, 70 official source-held-out test), four seeds (7, 17, 37, 42), optical misregistration tests, ablations and external GEOID-Flood events. The code in these notebooks has **not been re-executed or numerically reconciled** against every result in the submitted article.

## Suggested use

1. Upload the notebooks to Colab or open them directly from GitHub.
2. Review cells carefully before execution: some cells install packages, download files, connect to Drive, overwrite generated scripts, restore checkpoints, or launch expensive training. **Do not use Run all blindly.**
3. Restore the appropriate data, modules and checkpoints into the directory structure expected by each section, then run individual cells in sequence.
4. Before public release, verify dataset redistribution rights, review all URLs/paths and outputs for private information, and audit the paper's exact train/evaluation commands.

## Ethical and methodological notes

The official OMBRIA test set is source-held-out at patch level; geographic/event independence is not established by these files. GEOID-Flood transfer used a missing-post-event optical input adaptation. Statistical comparisons across four training seeds should be interpreted as exploratory.

## Citation

Please cite the final published paper when its authoritative bibliographic reference or DOI becomes available.
