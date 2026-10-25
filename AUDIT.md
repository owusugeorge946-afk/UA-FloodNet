# GitHub preparation audit

## File inspection

| Original | GitHub copy | Cells | Cells with cleared outputs | Original size | Cleaned size |
|---|---|---:|---:|---:|---:|
| `UA_FloodNet.ipynb` | `01_UA_FloodNet_Development.ipynb` | 81 | 80 | 10.60 MB | 0.20 MB |
| `UA_FloodNET_2.ipynb` | `02_UA_FloodNet_Training_and_Figures.ipynb` | 47 | 45 | 19.87 MB | 0.34 MB |
| `UA_flood_Net3.ipynb` | `03_UA_FloodNet_Ablations_Benchmarks_External.ipynb` | 42 | 42 | 1.97 MB | 0.40 MB |

## Changes made

- All original code and Markdown cell source preserved **verbatim**; notebooks were **not merged or refactored**, so scientific workflows and training choices remain unchanged.
- Removed embedded saved outputs and execution counts to reduce repository size and avoid publishing potentially private execution output.
- Removed stored widget state and transient cell execution metadata.
- Validated notebooks using `nbformat`.
- Extracted `train_multiseed.py` from a `%%writefile` cell, without editing its logic, and checked Python syntax.
- Added README and `.gitignore`.

## Outstanding items — verify before describing code as reproducible

1. Required modules `ablation_model.py`, `data.py`, `train_v4.py` and other local scripts, checkpoints, data archives and conversion metadata are absent from the three uploads.
2. Many cells rely on Colab-only `%` / `!` magics, Google Drive mounts and author-specific `/content/...` paths.
3. The notebooks were **not executed** in this environment, and dependency versions are not reliably pinned.
4. There are multiple exploratory or superseded training cells, so blindly running notebooks from top to bottom can run unintended jobs or overwrite files.
5. A pattern scan of cell source found no obvious literal API token or password, **not** a security guarantee; manually review URLs, checkpoints, author-specific paths and git history before publishing.
6. Verify numeric consistency and data split provenance before asserting exact article reproducibility.

## Release recommendation

Upload this package as an **archived research workflow**. To make it a complete reproducibility release, additionally supply audited source modules, pinned dependencies, tested setup, data provenance instructions, and model/checkpoint availability information.
