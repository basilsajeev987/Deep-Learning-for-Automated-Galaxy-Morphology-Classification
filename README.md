 A deep learning research project for automated galaxy morphology classification using **9 CNN architectures**, a large-scale astronomical image dataset, and an **NVIDIA A100-SXM4-40GB GPU**.

---

## 🌌 Project Overview

**COSMOS-AI** is an AI-powered computer vision project that investigates the use of deep convolutional neural networks to automatically classify galaxies according to their visual morphology.

The project formulates galaxy morphology as a binary image-classification problem:

```text
Astronomical Galaxy Image
          │
          ▼
    Deep CNN Model
          │
          ▼
 ┌─────────────────────┐
 │                     │
 │  Smooth             │
 │  Features / Disk    │
 │                     │
 └─────────────────────┘
```

The project was developed as a large-scale comparative experiment rather than a single-model classification task.

**Nine different CNN architectures** were trained and benchmarked using a common canonical dataset and evaluation protocol.

The complete training and experimentation workflow took approximately **two days** using an **NVIDIA A100-SXM4-40GB GPU** through Google Colab.

The final canonical validation benchmark selected:

> 🥇 **RegNetY-320**

as the best-performing architecture.

---

# 📌 Project Highlights

| Category | Details |
|---|---|
| Project | **COSMOS-AI** |
| Task | Automated galaxy morphology classification |
| Problem | Binary image classification |
| Classes | Smooth / Features-Disk |
| Original records | **209,294** |
| Usable galaxies | **136,997** |
| Training set | **95,897** |
| Validation set | **20,550** |
| Test set | **20,550** |
| CNN architectures | **9** |
| Training/experimentation | **~2 days** |
| GPU | **NVIDIA A100-SXM4-40GB** |
| GPU memory | ~39.49 GB |
| Image resolution | **224 × 224** |
| Optimizer | AdamW |
| Learning rate | **3 × 10⁻⁴** |
| Effective batch size | **256** |
| Mixed precision | CUDA AMP |
| Final model | **RegNetY-320** |
| Explainability | **Grad-CAM** |
| Environment | Google Colab |

---

# 🎯 Objectives

COSMOS-AI was developed with the following objectives:

1. Construct a high-confidence galaxy morphology dataset.
2. Create a reproducible train/validation/test split.
3. Train nine different CNN architectures.
4. Compare architectures using multiple performance metrics.
5. Select the strongest model using the canonical validation dataset.
6. Evaluate the selected model on an independent test dataset.
7. Analyse model errors.
8. Investigate model attention using Grad-CAM.
9. Establish a foundation for future AI-assisted astronomical research.

---

# 🪐 Dataset

The prepared source dataset contained:

| Property | Value |
|---|---:|
| Original dataset records | **209,294** |
| Original columns | **234** |
| Unique galaxies | **209,294** |
| Usable galaxies | **136,997** |
| Features/Disk | **88,429** |
| Smooth | **48,568** |
| Features/Disk proportion | **64.55%** |
| Smooth proportion | **35.45%** |

The final classification dataset contains only high-confidence examples.

---

# 🔬 Galaxy Classes

## Class 0 — Smooth

A galaxy is assigned the Smooth label when:

```text
t01_smooth_or_features_a01_smooth_debiased >= 0.80
```

Label:

```text
0
```

## Class 1 — Features/Disk

A galaxy is assigned the Features/Disk label when:

```text
t01_smooth_or_features_a02_features_or_disk_debiased >= 0.80
```

Label:

```text
1
```

Objects that do not satisfy either high-confidence condition are excluded.

This produces:

```text
136,997 usable galaxies
```

---

# 📊 Canonical Dataset Split

A fixed stratified split was used with:

```text
random_state = 42
```

The final split is:

| Dataset | Total | Smooth | Features/Disk |
|---|---:|---:|---:|
| Training | **95,897** | 33,997 | 61,900 |
| Validation | **20,550** | 7,285 | 13,265 |
| Test | **20,550** | 7,286 | 13,264 |
| **Total** | **136,997** | | |

The canonical files are:

```text
/content/drive/MyDrive/cosmos_ai/cosmos_canonical_train.csv
/content/drive/MyDrive/cosmos_ai/cosmos_canonical_val.csv
/content/drive/MyDrive/cosmos_ai/cosmos_canonical_test.csv
```

Using fixed canonical files ensures that all nine CNN architectures are evaluated on exactly the same validation population.

---

# 🖼️ Image Dataset

The astronomical image archive used by the project is:

```text
images_gz2.zip
```

Runtime extraction:

```text
/content/cosmos_images
```

Images are associated with galaxy records using `asset_id`.

Typical image path:

```text
/content/cosmos_images/images/{asset_id}.jpg
```

The relationship is:

```text
Galaxy record
     │
     └── asset_id
             │
             ▼
      Galaxy image (.jpg)
```

---

# 🧠 9 CNN Architectures

The core of COSMOS-AI is a benchmark involving **nine CNN architectures**.

## 1. ResNet-18

A lightweight residual network used as an efficient baseline.

## 2. ResNet-34

A deeper residual architecture used to investigate the effect of increased network depth.

## 3. ResNet-50

A widely used bottleneck residual CNN.

## 4. ResNet-101

A deeper residual architecture designed to investigate whether additional depth improves morphology classification.

## 5. DenseNet-169

A densely connected CNN that encourages feature reuse across layers.

## 6. EfficientNet-B4

A CNN using compound scaling of network depth, width and representation.

## 7. ConvNeXt-Tiny

A modern convolutional architecture based on contemporary computer-vision design principles.

## 8. RegNetY-320

The **best-performing architecture** in the final canonical validation benchmark.

## 9. SENet-154

A squeeze-and-excitation CNN architecture included in the comparative experiment.

---

# 🔎 Architecture Families

The nine models represent several different CNN design philosophies:

```text
ResNet
  └── Residual learning

DenseNet
  └── Dense feature reuse

EfficientNet
  └── Compound scaling

ConvNeXt
  └── Modern CNN architecture

RegNet
  └── Systematic network design

SENet
  └── Channel attention
```

This makes the experiment a genuine architecture benchmark rather than a single-model demonstration.

---

# ⏱️ Training and Experimentation

The complete model training and experimentation phase took approximately:

# **2 DAYS**

The project was developed and executed using:

```text
Google Colab
+
NVIDIA A100-SXM4-40GB
```

The A100 GPU was used for:

- CNN training
- Validation inference
- Model benchmarking
- Final model evaluation
- Error analysis
- Grad-CAM generation

---

# 🚀 NVIDIA A100 GPU

Primary hardware:

```text
NVIDIA A100-SXM4-40GB
```

Available memory:

```text
Approximately 39.49 GB
```

GPU optimisations included:

```python
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

The project also uses CUDA automatic mixed precision:

```python
torch.amp.autocast(...)
torch.amp.GradScaler(...)
```

These optimisations improve GPU memory efficiency and training throughput.

---

# ⚙️ Training Configuration

| Parameter | Configuration |
|---|---:|
| Random seed | **42** |
| Input resolution | **224 × 224** |
| Maximum epochs | **20** |
| Early stopping patience | **5** |
| Learning rate | **3 × 10⁻⁴** |
| Weight decay | **1 × 10⁻⁴** |
| Effective batch size | **256** |
| Optimizer | **AdamW** |
| Scheduler | **CosineAnnealingLR** |
| Loss | **CrossEntropyLoss** |
| Mixed precision | **Enabled** |
| Channels-last | **Enabled** |
| DataLoader workers | **8** |
| Prefetch factor | **4** |
| Pin memory | **Enabled** |
| Persistent workers | **Enabled** |

---

# 🔄 Data Augmentation

Training images use:

```text
RandomHorizontalFlip(p=0.5)
RandomVerticalFlip(p=0.5)
RandomRotation(15°)
```

followed by:

```text
Resize(224 × 224)
ToTensor()
Normalization
```

ImageNet normalization:

```text
Mean = [0.485, 0.456, 0.406]

Std  = [0.229, 0.224, 0.225]
```

Validation and test images use deterministic preprocessing without random augmentation.

---

# 📦 Effective Batch Size

The target effective batch size is:

```text
256
```

Gradient accumulation is used when the selected micro-batch is smaller:

```text
MICRO_BATCH_SIZE × ACCUMULATION_STEPS = 256
```

This allows the training configuration to maintain a consistent effective batch size.

---

# 🔬 Complete Deep Learning Pipeline

```text
                  SOURCE GALAXY DATASET
                           │
                           ▼
              MORPHOLOGY PROBABILITIES
                           │
                           ▼
                HIGH-CONFIDENCE FILTER
                           │
                           ▼
                  BINARY DATASET
                  /                              /                               ▼                 ▼
            SMOOTH          FEATURES/DISK
              (0)                 (1)
                 \               /
                  \             /
                   ▼           ▼
                STRATIFIED SPLIT
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
        TRAIN        VALIDATION     TEST
          │
          ▼
   IMAGE PREPROCESSING
          │
          ▼
    DATA AUGMENTATION
          │
          ▼
     9 CNN ARCHITECTURES
          │
          ▼
       TRAINING
          │
          ▼
 CANONICAL VALIDATION BENCHMARK
          │
          ▼
     MODEL COMPARISON
          │
          ▼
     FINAL MODEL
          │
          ▼
   INDEPENDENT TEST
          │
          ▼
    ERROR ANALYSIS
          │
          ▼
       GRAD-CAM
          │
          ▼
 SCIENTIFIC INTERPRETATION
```

---

# 📈 Evaluation Metrics

The project evaluates every architecture using multiple metrics.

### Accuracy

Overall proportion of correctly classified galaxies.

### Precision

Proportion of predicted Features/Disk galaxies that are actually Features/Disk.

### Recall

Proportion of actual Features/Disk galaxies correctly detected.

### Specificity

Proportion of Smooth galaxies correctly identified.

### F1-score

Harmonic mean of precision and recall.

### Balanced Accuracy

Average of recall and specificity.

### ROC-AUC

Measures ranking performance across classification thresholds.

### PR-AUC

Measures precision-recall performance and is especially informative for imbalanced classification.

---

# 🏆 9-Model Canonical Validation Results

| Rank | Model | Accuracy | Precision | Recall | Specificity | F1 | Balanced Acc. | ROC-AUC | PR-AUC |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 🥇 1 | **RegNetY-320** | **92.998%** | 95.062% | 94.037% | 91.105% | **94.547%** | 92.571% | **98.212%** | **99.095%** |
| 🥈 2 | EfficientNet-B4 | 92.730% | 94.577% | 94.135% | 90.172% | 94.355% | 92.153% | 98.154% | 99.042% |
| 3 | DenseNet-169 | 92.720% | 95.521% | 93.087% | 92.052% | 94.288% | 92.570% | 98.156% | 99.054% |
| 4 | ResNet-101 | 92.676% | 95.440% | 93.102% | 91.901% | 94.257% | 92.502% | 98.128% | 99.031% |
| 5 | ResNet-50 | 92.365% | 94.757% | 93.336% | 90.597% | 94.041% | 91.966% | 97.973% | 98.958% |
| 6 | ResNet-18 | 92.345% | 95.388% | 92.620% | 91.846% | 93.984% | 92.233% | 97.925% | 98.930% |
| 7 | ResNet-34 | 92.263% | 94.375% | 93.592% | 89.842% | 93.982% | 91.717% | 97.979% | 98.965% |
| 8 | ConvNeXt-Tiny | 92.209% | 94.289% | 93.600% | 89.677% | 93.943% | 91.639% | 97.913% | 98.917% |
| 9 | SENet-154 | 36.161% | 54.926% | 6.136% | 90.830% | 11.040% | 48.483% | 42.367% | 59.595% |

---

# 🥇 Final Model Selection

## RegNetY-320

The final selected model is:

> **RegNetY-320**

It achieved the highest canonical validation performance across the benchmark.

### Validation Performance

```text
Accuracy          : 92.998%
Precision         : 95.062%
Recall            : 94.037%
Specificity       : 91.105%
F1                : 94.547%
Balanced Accuracy : 92.571%
ROC-AUC           : 98.212%
PR-AUC            : 99.095%
```

Checkpoint:

```text
/content/drive/MyDrive/cosmos_ai/cnn_models/regnety_320_best.pth
```

Parameters:

```text
141,341,196
```

Validation confusion matrix:

```text
TN = 6637
FP =  648
FN =  791
TP = 12474
```

Inference performance:

```text
Inference time : 19.655 seconds
Throughput     : 1045.55 images/second
Peak GPU usage : 8.044 GB
```

---

# 🥈 Runner-Up

The second-best model was:

```text
EfficientNet-B4
```

### F1 comparison

```text
RegNetY-320     = 0.945466
EfficientNet-B4 = 0.943554

Difference      = 0.001911
```

### Accuracy comparison

```text
RegNetY-320     = 0.929976
EfficientNet-B4 = 0.927299

Difference      = 0.002676
```

---

# 🧪 Final Test Evaluation

After completing the nine-model canonical validation benchmark, RegNetY-320 was selected for independent test evaluation.

Canonical test dataset:

```text
20,550 galaxies
```

Class distribution:

```text
Smooth        = 7,286
Features/Disk = 13,264
```

Final test artifacts:

```text
FINAL_MODEL_CANONICAL_TEST_RESULTS.csv
FINAL_MODEL_CANONICAL_TEST_PREDICTIONS.csv
```

Location:

```text
/content/drive/MyDrive/cosmos_ai/cnn_benchmark_results/
```

The final test metrics should be taken directly from the generated test-results CSV when preparing a final report.

---

# 🔍 Error Analysis

The final model is analysed using:

- Confusion matrix
- ROC curve
- Precision-recall curve
- False positives
- False negatives
- Low-confidence predictions
- Confidence distributions
- Class-specific error rates

Generated files include:

```text
FINAL_MODEL_confusion_matrix.png
FINAL_MODEL_ROC_curve.png
FINAL_MODEL_PR_curve.png
FINAL_MODEL_low_confidence_samples.csv
FINAL_MODEL_false_positives.csv
FINAL_MODEL_false_negatives.csv
FINAL_MODEL_confidence_analysis.csv
FINAL_MODEL_ERROR_ANALYSIS_SUMMARY.csv
```

---

# 👁️ Grad-CAM Explainability

Grad-CAM was applied to the final RegNetY-320 model.

Target layer:

```text
s4.b1.downsample.conv
```

The analysis included:

```text
3 Correct Smooth
3 Correct Features/Disk
3 False Positives
3 False Negatives
```

Selected false positives were extremely confident:

```text
P(Features/Disk) = 0.9985
P(Features/Disk) = 0.9953
P(Features/Disk) = 0.9941
```

Selected false negatives were also highly confident:

```text
Smooth confidence = 0.9734
Smooth confidence = 0.9681
Smooth confidence = 0.9654
```

This demonstrates that some difficult galaxies are high-confidence misclassifications rather than simply uncertain predictions.

Grad-CAM results:

```text
/content/drive/MyDrive/cosmos_ai/cnn_benchmark_results/gradcam_final_model/
```

Grad-CAM should be interpreted as showing image regions influential to the model prediction. It does not by itself prove that the model learned scientifically correct or causal morphology features.

---

# 📁 Project Outputs

### Model checkpoints

```text
*_best.pth
```

### Validation results

```text
*_canonical_validation_results.csv
```

### Validation predictions

```text
*_canonical_validation_predictions.csv
```

### Final model comparison

```text
COSMOS_AI_9_MODEL_FINAL_COMPARISON.csv
```

### Comparison plots

```text
COSMOS_AI_model_comparison_F1.png
COSMOS_AI_model_comparison_accuracy.png
COSMOS_AI_model_comparison_ROCAUC.png
COSMOS_AI_model_comparison_PRAUC.png
```

### Final test

```text
FINAL_MODEL_CANONICAL_TEST_RESULTS.csv
FINAL_MODEL_CANONICAL_TEST_PREDICTIONS.csv
```

### Error analysis

```text
FINAL_MODEL_confusion_matrix.png
FINAL_MODEL_ROC_curve.png
FINAL_MODEL_PR_curve.png
FINAL_MODEL_low_confidence_samples.csv
FINAL_MODEL_false_positives.csv
FINAL_MODEL_false_negatives.csv
FINAL_MODEL_confidence_analysis.csv
FINAL_MODEL_ERROR_ANALYSIS_SUMMARY.csv
```

### Explainability

```text
gradcam_final_model/
gradcam_sample_index.csv
```

---

# 🗂️ Project Structure

```text
cosmos_ai/
│
├── cosmos_canonical_train.csv
├── cosmos_canonical_val.csv
├── cosmos_canonical_test.csv
│
├── cnn_models/
│   ├── resnet18_best.pth
│   ├── resnet34_best.pth
│   ├── resnet50_best.pth
│   ├── resnet101_best.pth
│   ├── efficientnet_b4_best.pth
│   ├── convnext_tiny_best.pth
│   ├── regnety_320_best.pth
│   └── senet154_best.pth
│
├── densenet169_best.pth
│
├── cnn_benchmark_results/
│   ├── *_canonical_validation_results.csv
│   ├── *_canonical_validation_predictions.csv
│   ├── COSMOS_AI_9_MODEL_FINAL_COMPARISON.csv
│   ├── COSMOS_AI_model_comparison_F1.png
│   ├── COSMOS_AI_model_comparison_accuracy.png
│   ├── COSMOS_AI_model_comparison_ROCAUC.png
│   ├── COSMOS_AI_model_comparison_PRAUC.png
│   ├── FINAL_MODEL_CANONICAL_TEST_RESULTS.csv
│   ├── FINAL_MODEL_CANONICAL_TEST_PREDICTIONS.csv
│   ├── FINAL_MODEL_confusion_matrix.png
│   ├── FINAL_MODEL_ROC_curve.png
│   ├── FINAL_MODEL_PR_curve.png
│   ├── FINAL_MODEL_low_confidence_samples.csv
│   ├── FINAL_MODEL_false_positives.csv
│   ├── FINAL_MODEL_false_negatives.csv
│   ├── FINAL_MODEL_confidence_analysis.csv
│   ├── FINAL_MODEL_ERROR_ANALYSIS_SUMMARY.csv
│   └── gradcam_final_model/
│
└── ...
```

---

# 🛠️ Technology Stack

## Programming

```text
Python
```

## Deep Learning

```text
PyTorch
torchvision
timm
```

## Data Processing

```text
NumPy
Pandas
scikit-learn
```

## Computer Vision

```text
Pillow
Torchvision Transforms
Grad-CAM
```

## Hardware Acceleration

```text
CUDA
NVIDIA A100-SXM4-40GB
CUDA AMP
TF32
```

## Development Environment

```text
Google Colab
Google Drive
Jupyter / Colab Notebooks
```

---

# 🔁 Reproducibility

The project uses:

```text
Random seed = 42
```

The canonical dataset files are saved and reused across the benchmark.

All nine CNN architectures are therefore evaluated using the same canonical validation and test populations.

The benchmark maintains consistent:

```text
Image resolution
Normalization
Augmentation
Optimizer
Learning rate
Weight decay
Effective batch size
Evaluation protocol
```

This makes the architecture comparison more reproducible and scientifically meaningful.

---

# ⚠️ Historical Split Warning

An earlier experimental split existed during development:

```text
Train      = 167,701
Validation = 35,936
Test       = 35,936
```

This is **not** the final benchmark split.

The final COSMOS-AI benchmark uses:

```text
Train      = 95,897
Validation = 20,550
Test       = 20,550
```

Historical metrics from the earlier split should not be mixed with the final canonical benchmark.

---

# ⚠️ Limitations

### Binary classification

The current project classifies galaxies into only two categories:

```text
Smooth
Features/Disk
```

Real galaxy morphology is substantially more complex.

### Class imbalance

The Features/Disk class contains more galaxies than the Smooth class.

### Label thresholding

The 0.80 threshold increases confidence but excludes ambiguous examples.

### Dataset bias

The model may learn correlations specific to the source dataset or imaging pipeline.

### Image resolution

Models currently operate at:

```text
224 × 224
```

Higher resolution could preserve additional morphological information.

### Explainability

Grad-CAM provides visual evidence of influential regions but does not establish scientific causality.

### SENet-154

SENet-154 produced a major performance outlier compared with the other models. This result should be independently investigated before making broad architectural conclusions.

---

# 🚀 Future Work

Potential extensions include:

- Multi-class galaxy morphology classification
- Higher-resolution image training
- Vision Transformers
- Swin Transformer / Swin V2
- Self-supervised astronomical representation learning
- Model calibration
- Uncertainty estimation
- External astronomical dataset validation
- Integrated Gradients
- Occlusion analysis
- Feature embedding analysis
- Nearest-neighbour galaxy retrieval
- Large-scale astronomical survey analysis
- AI-assisted galaxy evolution research

---

# 🔭 Research Significance

COSMOS-AI demonstrates a complete research-oriented deep learning workflow:

```text
Dataset Construction
        ↓
High-Confidence Labeling
        ↓
Canonical Data Split
        ↓
Image Preprocessing
        ↓
Data Augmentation
        ↓
9 CNN Architectures
        ↓
~2 Days of Training
        ↓
NVIDIA A100 GPU
        ↓
Canonical Validation Benchmark
        ↓
Final Model Selection
        ↓
Independent Test
        ↓
Error Analysis
        ↓
Grad-CAM Explainability
        ↓
Scientific Interpretation
```

The project combines:

```text
Artificial Intelligence
Computer Vision
Deep Learning
Astronomy
Astrophysics
Cosmology
Explainable AI
```

It provides a foundation for future research into AI-assisted astronomical image analysis and potentially larger-scale galaxy morphology and galaxy-evolution studies.

---

# 📋 Final Project Summary

```text
============================================================
COSMOS-AI
Deep Learning for Automated Galaxy Morphology Classification
============================================================

Original records
----------------
209,294

Usable galaxies
---------------
136,997

Classes
-------
0 = Smooth
1 = Features/Disk

Canonical dataset
-----------------
Train      : 95,897
Validation : 20,550
Test       : 20,550

CNN architectures
-----------------
9

1. ResNet-18
2. ResNet-34
3. ResNet-50
4. ResNet-101
5. DenseNet-169
6. EfficientNet-B4
7. ConvNeXt-Tiny
8. RegNetY-320
9. SENet-154

Training duration
-----------------
Approximately 2 days

GPU
---
NVIDIA A100-SXM4-40GB

Image size
----------
224 × 224

Final model
-----------
RegNetY-320

Validation Accuracy
-------------------
92.998%

Validation F1
-------------
94.547%

Validation ROC-AUC
------------------
98.212%

Validation PR-AUC
-----------------
99.095%

Explainability
--------------
Grad-CAM
```

---

# 👨‍💻 Author

**Basil Sajeev**

MSc Computer Science  
Ulster University, Manchester Campus

Interests:

```text
Artificial Intelligence
Machine Learning
Deep Learning
Computer Vision
AI Agents
Astrophysics
Cosmology
Astronomical AI
```

---

# 📄 License

Add an appropriate license before publishing the repository.

The original astronomical dataset and image data may have separate licensing and usage requirements. Verify the original dataset terms before redistributing the data.

---

## COSMOS-AI

### **Deep Learning for Automated Galaxy Morphology Classification**
