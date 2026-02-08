<img width="1902" height="310" alt="image" src="https://github.com/user-attachments/assets/1beffec0-7cd0-44af-abb6-475701c64943" />

# Semiconductor Defect Detection – Edge AI Pipeline (Phase 1)

This repository contains an **end-to-end edge-ready defect classification pipeline for SEM (Scanning Electron Microscope) images**

The solution focuses on:
- Grayscale SEM image understanding  
- Data scarcity handling via synthetic generation  
- Edge deployment using ONNX Runtime  
- Model compression via **knowledge distillation**

---

## Problem Statement

Manual inspection of semiconductor wafers using SEM images is:
- Time-consuming
- Operator-dependent
- Not scalable for high-volume fabs

The goal is to **automatically classify SEM images into defect categories** on **edge devices**, ensuring:
- High accuracy
- Low latency
- Small model size
- Robustness to noise and imaging variations

---

## Proposed Solution 

1. **Base Dataset (Grayscale SEM)**
   - Manually labeled defect images
   - Limited and imbalanced initially

2. **Synthetic Dataset Generation**
   - Controlled noise, blur, distortion
   - Class balancing (equal samples per class)
   - Improves generalization

3. **Teacher Model**
   - ConvNeXt (grayscale input)
   - High-capacity, high-accuracy model

4. **Knowledge Distillation**
   - Teacher → Student transfer
   - Preserves accuracy while reducing size

5. **Student Model (Edge-Ready)**
   - MobileNetV3 (grayscale, lightweight)
   - Optimized for CPU inference

6. **ONNX Export & Edge Inference**
   - PIL-based preprocessing (grayscale-safe)
   - ONNX Runtime (CPU)
   - Semantic guard for reliability (recently put to 0)

---

##  Dataset Plan & Class Design

- **Total Images (after synthesis):** ~300 per class = 2400 images
- **Number of Classes:** 8  
  - `clean`
  - `bridge`
  - `cmp`
  - `crack`
  - `open`
  - `ler`
  - `via`
  - `other`
- **Image Type:** Grayscale SEM images
- **Labeling Method:** Manual + synthetic generation
- **Train / Validation / Test Split:** 70 / 15 / 15
- **Class Balance:** Equal samples per class

---

##  Model Architecture (Phase 1 Baseline)
<img width="1212" height="500" alt="semicon drawio" src="https://github.com/user-attachments/assets/2866b1c6-9538-4cf5-9732-53614435814d" />


### Teacher Model
- Architecture: ConvNeXt
- Input: 1 × 224 × 224 (grayscale)
- Training: Transfer learning 
- Purpose: High-accuracy feature extraction

### Student Model (Final)
- Architecture: MobileNetV3
- Input: 1 × 224 × 224 (grayscale)
- Training: Knowledge Distillation
- Deployment: Edge CPU (ONNX Runtime)

---

##  Phase-1 Results (Student Model)

Evaluation performed on a **mixed test set** (clean + noisy SEM images).

- **Overall Accuracy:** 84.46%(on noise test dataset) else 98% to favourable
- **Strong performance on:** clean, via, other
- **Robust to noise & blur introduced during synthesis (available in 2test.zip)**

## Confusion matrix and class-wise accuracy are available evaluation_results.json and evaluation_results.png in ROOT DIRECTORY.

---

##  Edge Inference Pipeline

**Preprocessing**
- PIL grayscale loading
- Resize → 224 × 224
- Normalize to [0, 1]
- Shape: `[1, 1, 224, 224]`

**Inference**
- ONNX Runtime (CPU)
- MobileNetV3 (ONNX)

**Post-processing**
- Softmax + confidence threshold
- Semantic guard for ambiguous defects
- Final defect classification

---
# Some Important links
- Dataset DRIVE LINK(test zip file also provided) : ```https://drive.google.com/drive/folders/16eWdfwJfuVV2kBwTZAlm7GFnwZbKSBzu?
usp=sharing```
- Model (.pth) + ONNX Link : ```https://drive.google.com/drive/folders/1Sx_sw75ysi-lVEnsodQMJ64DUIs-huav```
- Evaluation report and Confusion Matrix : ROOT DIRECTORY
---
## How To Run
- clone reposittory :
  ```
  https://github.com/haddybhaiya/sem-i-con.git
   ```
- Download requirements:
  ```
  pip install -r requirements.txt
   ```
- Download and paste :
   - models (FP32) in ```/models```
   - dataset in ```/dataset```
- Run phase1_eval.py
  ```
  py evaluation/phase1_eval.py
  ```
  
     


## Repository Structure

```text
.
├── training/
│   └── model.py                # MobileNetV3 / ConvNeXt definitions
│
├── edge/
│   ├── edge_infer.py           # Single image inference
│   ├── auto_edge.py            # Adaptive edge inference
│   ├── export_onnx.py          # PyTorch → ONNX export
│
├── evaluation/
│   ├── create_test_set.py
│   ├── phase1_eval.py
│   ├── metrics.py
│   └── report.json
│
├── dataset/
│   ├── synthetic_dataset/
│   └── test/
│
├── models/
│   ├── mobilenetv3_sem_distilled.pth (not pushed)
│   └── mobilenetv3_sem.onnx (not pushed)
│
└── README.md

```
