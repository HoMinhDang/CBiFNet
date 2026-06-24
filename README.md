# CBiF-Net: A Lightweight CNN-Transformer Network with Bilinear Feature Fusion for Crack Segmentation

[![Static Badge](https://img.shields.io/badge/Paper-Springer_LNCS-red)](https://link.springer.com/chapter/10.1007/978-981-92-0068-9_6) &nbsp;
[![Static Badge](https://img.shields.io/badge/Weight-Google_Drive-blue)](https://drive.google.com/drive/folders/1T_Y8zEUmDXFqBi57kfQRttxc7P31wJ0l?usp=sharing)

This repo contains the PyTorch implementation of CBiF-Net for pixel-wise crack segmentation in civil infrastructure images.

## 1. Model Architecture

![](/Image/Model.png)

*Overall architecture of CBiF-Net. The framework comprises a CNN branch (MobileNetV3-Large) for local feature extraction and a Transformer branch (Mix Transformer) for global context modeling, fused via BiFusion modules at each stage.*

- Trained Model: [weight](https://drive.google.com/drive/folders/1T_Y8zEUmDXFqBi57kfQRttxc7P31wJ0l?usp=sharing)

### 1.1 BiFusion Module

![](/Image/Bifusion_module.png)

*The Bi-linear Interaction Module (BiFusion) models cross-modality correlations via element-wise multiplication, then aggregates raw CNN features, raw Transformer features, and the interaction map through a residual bottleneck block.*

## 2. Dataset

We evaluate on three public crack segmentation benchmarks:

| Dataset | Description | Size | Resolution |
|---------|-------------|------|------------|
| **DeepCrack** | Pavement crack images | 537 | 544×384 |
| **SteelCrack** | Metallic & steel surface cracks | 4,355 | 256×256 |
| **CrackVision12K** | Pavement & masonry with complex textures | 12,000 | 256×256 |

![](/Image/Datasets.jpg)

*Visual examples: DeepCrack (left), SteelCrack (middle), CrackVision12K (right).*

Data organized as:

```text
<DATASET_ROOT>/
  train/
    IMG/
    GT/
  val/
    IMG/
    GT/
  test/
    IMG/
    GT/
```

Configure the root path in `CBiF-Net/config.py`:

```python
dataset = "./data/"  # update to your local path
```

## 3. Installation

```bash
git clone <repo-url>
cd CBiFNet
pip install -r requirements.txt
```

## 4. Usage

### File Structure

- `CBiF-Net/model.py` – CBiF-Net architecture
- `CBiF-Net/trainer.py` – training loop
- `CBiF-Net/test.py` – testing / evaluation
- `CBiF-Net/dataloader.py`, `dataset.py` – data loading utilities
- `CBiF-Net/config.py` – global configuration
- `CBiF-Net/metric.py` – evaluation metrics
- `CBiF-Net/utils.py` – helper functions

### Train

```bash
python -m CBiF-Net.trainer
# or
python CBiF-Net/trainer.py
```

Main training configs (epochs, batch size, learning rate, etc.) are defined in `CBiF-Net/config.py`.

### Test / Evaluation

```bash
python CBiF-Net/test.py
```

## 5. Results

### Comparison with State-of-the-Art

| Dataset | Method | P (%) | R (%) | F1 (%) | mIoU (%) |
|---------|--------|------:|------:|-------:|---------:|
| **DeepCrack** | U-Net | 87.07 | 74.96 | 80.56 | 78.43 |
| | SegFormer-B0 | 85.26 | 83.41 | 84.32 | 76.94 |
| | HybridSegmentor | 88.58 | 83.38 | 85.90 | 78.08 |
| | **CBiF-Net (Ours)** | **89.20** | **87.59** | **88.39** | **79.32** |
| **SteelCrack** | U-Net | 91.90 | 84.68 | 88.91 | 80.03 |
| | SegFormer-B0 | 86.78 | 89.94 | 88.33 | 79.10 |
| | HybridSegmentor | 91.79 | 86.29 | 88.95 | 80.11 |
| | **CBiF-Net (Ours)** | **93.19** | **92.60** | **92.89** | **86.73** |
| **CrackVision12K** | U-Net | 79.20 | 72.40 | 75.60 | 60.80 |
| | SegFormer-B0 | 81.26 | 73.14 | 76.99 | 62.59 |
| | HybridSegmentor | 80.97 | 74.24 | 77.46 | 63.21 |
| | **CBiF-Net (Ours)** | 80.15 | **75.82** | **77.93** | **63.84** |

**Model Complexity (256×256 input)**

| Method | Params | GMACs | FPS |
|--------|-------:|------:|----:|
| U-Net | 7.8M | 13.75 | 152 |
| SegFormer-B0 | 3.32M | 0.54 | 77 |
| HybridSegmentor | 226.8M | 134.08 | 10 |
| **CBiF-Net (Ours)** | **5.74M** | **2.98** | **121** |

### Visual Comparison

![](/Image/compare_sota.jpg)

*Qualitative comparison with SOTA methods on SteelCrack (a–b), CrackVision12K (c–e), and DeepCrack (f–g).*

### Failure Cases

![](/Image/fail_case_conf.jpg)

*Failure cases: false negatives (red) and false positives (blue) caused by pen markings, watermarks, and shadows.*

## 6. Citation

If you use this code in your research, please cite our paper:

```bibtex
@InProceedings{10.1007/978-981-92-0068-9_6,
  author    = {Dang, Ho Minh and Thu, Be La Anh and Viet, Vo Hoai},
  editor    = {Nguyen, Ngoc Thanh and Chen, Chun-Hao and Fujita, Hamido and Hong, Tzung-Pei and Manolopoulos, Yannis and Wojtkiewicz, Krystian},
  title     = {{CBiF-Net}: A Lightweight {CNN-Transformer} Network with Bilinear Feature Fusion for Crack Segmentation},
  booktitle = {Recent Challenges in Intelligent Information and Database Systems},
  year      = {2026},
  publisher = {Springer Nature Singapore},
  address   = {Singapore},
  pages     = {71--85},
  isbn      = {978-981-92-0068-9}
}
```

If you have any questions, please contact `hmdang22@fit.hcmus.edu.vn` or `blathu22@fit.hcmus.edu.vn` without hesitation.
