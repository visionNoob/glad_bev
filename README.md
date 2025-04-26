# GLAD-BEV: Gridless and Adaptive Dynamic BEV Representation for Multi-Camera Object Detection

<p align="center">
  <img src="https://img.shields.io/badge/status-active-brightgreen?style=flat-square">
  <img src="https://img.shields.io/github/license/your-repo/glad-bev?style=flat-square">
  <img src="https://img.shields.io/github/stars/your-repo/glad-bev?style=flat-square">
  <img src="https://img.shields.io/github/issues/your-repo/glad-bev?style=flat-square">
  <img src="https://img.shields.io/github/languages/top/your-repo/glad-bev?style=flat-square">
</p>

---

> Official code for **GLAD-BEV**, proposed in our ICCV 2025 submission:  
> **"Gridless and Adaptive Dynamic BEV Representation for Multi-Camera Object Detection"**.

---

## ✨ Features
- Gridless dynamic BEV representation
- Cross-view feature aggregation without dense grids
- Transformer-based query refinement
- Strong generalization to unseen scenes (GMVD benchmark)
- Clean PyTorch Lightning-based training pipeline

---

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/your-repo/glad-bev.git
cd glad-bev

# (Optional) Create a virtual environment
conda create -n gladbev python=3.10
conda activate gladbev

# Install dependencies
pip install -r requirements.txt
```

---

## ⚡ Quick Start

### Train on WildTrack

```bash
python scripts/train.py --config configs/wildtrack.yaml
```

### Evaluate on GMVD unseen scenes

```bash
python scripts/val.py --config configs/gmvd_unseen.yaml
```

---

## 🏗️ Project Structure

```plaintext
glad_bev/
├── configs/                 # Training and evaluation config files
├── datasets/                # Data loaders for WildTrack, MultiviewX, GMVD
├── models/                  # GLAD-BEV model, backbone, transformer modules
├── lightning/               # PyTorch Lightning training modules
├── utils/                   # Projection, metrics utilities
├── scripts/                 # Training and validation scripts
├── figures/                 # Figures used in the paper
├── requirements.txt         # Python requirements
├── Dockerfile               # Docker environment setup
└── README.md                # This file
```

---

## 📄 Citation

If you find this work helpful, please consider citing:

```bibtex
@inproceedings{your2025gladbev,
  title={GLAD-BEV: Gridless and Adaptive Dynamic BEV Representation for Multi-Camera Object Detection},
  author={Your Name and Collaborators},
  booktitle={International Conference on Computer Vision (ICCV)},
  year={2025}
}
```

---

## 📜 License

This repository is released under the MIT License.

---

## 🤝 Acknowledgements

We thank the authors of BEVFormer, MVDet, and GMVD for their inspiring works.
