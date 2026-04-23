# K-STEMIT: Knowledge-Informed Spatio-Temporal Efficient Multi-Branch Graph Neural Network for Subsurface Stratigraphy Thickness Estimation from Radar Data

[![arXiv](https://img.shields.io/badge/arXiv-2604.09922-b31b1b.svg)](https://arxiv.org/abs/2604.09922)
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
@misc{liu2026kstemitknowledgeinformedspatiotemporalefficient,
  title={K-STEMIT: Knowledge-Informed Spatio-Temporal Efficient Multi-Branch Graph Neural Network for Subsurface Stratigraphy Thickness Estimation from Radar Data},
  author={Zesheng Liu and Maryam Rahnemoonfar},
  year={2026},
  eprint={2604.09922},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2604.09922}
}
```

## Contact

For questions about the code or the paper, please open an issue in this repository or contact the corresponding author at `maryam@lehigh.edu`.
