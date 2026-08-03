# K-STEMIT: Knowledge-Informed Spatio-Temporal Efficient Multi-Branch Graph Neural Network for Subsurface Stratigraphy Thickness Estimation from Radar Data

[![arXiv](https://img.shields.io/badge/2604.09922-B31B1B?label=arXiv&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.09922)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official code release for the paper **"K-STEMIT: Knowledge-Informed Spatio-Temporal Efficient Multi-Branch Graph Neural Network for Subsurface Stratigraphy Thickness Estimation from Radar Data"** ([arXiv:2604.09922](https://arxiv.org/abs/2604.09922)).

The manuscript is currently under review. This repository is shared to support reproducibility, open-source evaluation, and follow-up research on subsurface stratigraphy thickness estimation from radar data.

## Authors and Collaboration

- Author: Zesheng Liu
- Developed in collaboration with Maryam Rahnemoonfar's research group (Bina Lab) at Lehigh University

## Overview

This repository includes:

- K-STEMIT training code for knowledge-informed and non-knowledge-informed settings
- baseline GNN variants and ablation models
- standard, multi-dataset, and shared-split training runners
- a sample `tmux`-based training launcher

## Requirements

Please install the following dependencies before running the code:

- Python 3.10 or later
- PyTorch
- PyTorch Geometric
- PyTorch Geometric Temporal
- Hugging Face Accelerate
- NumPy
- `dill`
- `tqdm`
- `tmux` (optional, only for `K-STEMIT-train.sh`)

## Quick Start

`K-STEMIT-train.sh` provides a sample way to launch training in a `tmux` session. From the repository root, a typical run looks like:

```bash
CODE_DIR="$(pwd)" OUTPUT_DIR="$(pwd)/outputs" ./K-STEMIT-train.sh
```

You can also launch the main training entry point directly:

```bash
accelerate launch run_new_acc.py \
  --model Ablation1 \
  --batch 1 \
  --epoch 450 \
  --adaptive False \
  --ablation True \
  --featureablation 0101100 \
  --lr 5e-3 \
  --scheduler cosine \
  --schedulerargs 450 \
  --eta_min 1e-7 \
  --weight_decay 1e-5 \
  --folder Experiment
```

## Repository Layout

- `README.md`: project overview, setup notes, citation, and contact information
- `LICENSE`: MIT license for the repository
- `K-STEMIT-train.sh`: sample `tmux`-based launcher for a configurable training run
- `run_new_acc.py`: main training pipeline for the standard K-STEMIT setting
- `run_new_acc_baseline.py`: standalone baseline training pipeline
- `run_new_acc_multi.py`: multi-dataset training pipeline for `dataset_predict_xx` variants
- `run_new_acc_multi_shared21.py`: shared-source multi-dataset training pipeline with target cropping
- `model_mb.py`: knowledge-informed multi-branch K-STEMIT models and ablation variants
- `model_mb_nonphy.py`: multi-branch K-STEMIT variants without physical input features
- `model_fused.py`: fused single-branch GCN, GAT, and GraphSAGE baselines
- `model_baseline.py`: additional baseline architecture definitions
- `GAT_LSTM.py`: custom GAT-based recurrent graph cell used by model variants
- `SAGE_LSTM.py`: custom GraphSAGE-based recurrent graph cell used by model variants
- `utils.py`: dataset utilities, split generation, evaluation metrics, and checkpoint saving

## Data Availability

The processed dataset used in the paper is not bundled with this repository. Please contact us if you need access to the processed dataset for research purposes.

## License

This repository is released under the [MIT License](LICENSE).

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{liu2024multibranchspatiotemporalgraphneural,
      title={Multi-branch Spatio-Temporal Graph Neural Network For Efficient Ice Layer Thickness Prediction}, 
      author={Zesheng Liu and Maryam Rahnemoonfar},
      year={2024},
      eprint={2411.04055},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2411.04055}, 
}
```

```bibtex
@INPROCEEDINGS{11031955,
  author={Liu, Zesheng and Rahnemoonfar, Maryam},
  booktitle={2025 IEEE International Radar Conference (RADAR)}, 
  title={Physics-Informed Spatio-Temporal Graph Neural Network for Efficient Deep Ice Layer Thickness Estimation in Radar Imagery}, 
  year={2025},
  volume={},
  number={},
  pages={1-6},
  keywords={Radar remote sensing;Snow;Atmospheric modeling;Radar;Radar imaging;Radar tracking;Ice;Graph neural networks;Synchronization;Meteorology;Deep Learning;Physics-informed learning;Spatio-Temporal Learning;Graph Neural Network;Ice Layer;Ice Thickness;Remote Sensing},
  doi={10.1109/RADAR52380.2025.11031955}}

```

```bibtex
@article{LIU2026134633,
title = {K-STEMIT: Knowledge-informed spatio-temporal efficient multi-branch graph neural network for subsurface stratigraphy thickness estimation from radar data},
journal = {Neurocomputing},
pages = {134633},
year = {2026},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2026.134633},
url = {https://www.sciencedirect.com/science/article/pii/S092523122602031X},
author = {Zesheng Liu and Maryam Rahnemoonfar},
keywords = {Deep learning, Spatio-temporal learning, Knowledge-informed, Graph neural network, Ice thickness, Remote sensing, Radar stratigraphy},
abstract = {Spatio-temporal patterns in subsurface stratigraphy encode key information about accumulation, deformation, and layer formation processes. For polar ice sheets and corresponding subsurface ice layer stratigraphy, variations in layer thickness provide quantitative constraints that support snow mass balance estimation, improved projections of ice sheet change, and reduced uncertainty in climate and engineering models. Radar sensors capture these layered structures as depth-resolved radargrams; however, convolutional neural networks applied directly to radargrams are often sensitive to speckle noise and acquisition artifacts. More broadly, purely data-driven approaches tend to underuse known physical knowledge, which can produce inconsistent or unrealistic thickness estimates when extrapolating across space and time. To address these challenges, we develop K-STEMIT, a novel knowledge-informed, efficient, multi-branch spatio-temporal graph neural network that combines a geometric framework for spatial learning with temporal convolution to capture temporal dynamics, and incorporates physical data synchronized from the Model Atmospheric Regional physical weather model. An adaptive feature fusion strategy is employed to dynamically combine features learned from different branches. Extensive experiments on a standardized snow-radar benchmark are conducted to compare K-STEMIT against current state-of-the-art methods in both knowledge-informed and non-knowledge-informed settings, as well as other existing methods. Results show that K-STEMIT consistently achieves the highest accuracy while maintaining near-optimal efficiency. Most notably, incorporating adaptive feature fusion and physical priors reduces the root mean-squared error by 21.99% with negligible additional cost compared to its conventional multi-branch variants. Additionally, our proposed K-STEMIT achieves consistently lower per-year relative MAE, supporting reliable, continuous spatiotemporal assessment of snow accumulation variability across large spatial regions.}
}
```

## Contact

For questions about the code or the paper, please open an issue in this repository or contact the corresponding author at `maryam@lehigh.edu`.
