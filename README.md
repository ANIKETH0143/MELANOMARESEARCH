# MELANOMARESEARCH

Research code and verified experimental results for an explainable, fairness-aware and uncertainty-aware skin lesion classification framework.

## Research Pipeline

Preprocessing -> Segmentation -> Classification -> Explainability -> Fairness -> Uncertainty -> Clinical Validation -> Statistical Comparison

## Classification Models

The repository contains implementations and benchmark results for:

- DenseNet121
- ResNet101
- EfficientNetV2
- ConvNeXt
- Swin Transformer
- DermaFair-XFormer (proposed framework)

## Datasets

Experiments use:

- ISIC2018
- HAM10000
- PH2

PH2 is treated as an independent external evaluation dataset in the planned experimental framework.

## Segmentation

A U-Net based lesion segmentation pipeline is implemented using the paired ISIC2018 segmentation images and masks.

## Verified Results

Verified benchmark results and analysis files are available under:

`results/`

The repository also contains the current PH2 optimized Swin Transformer evaluation and error analysis.

## Reproducibility

Raw datasets and trained model checkpoint files are intentionally excluded from the repository.

Dataset split information, research configuration, source code, literature audit files, and verified results are included where appropriate.

## Status

This repository contains the current research implementation and verified experimental results. Components that require additional experimentation, including full segmentation training, explainability, fairness, uncertainty, clinical validation, and statistical comparison, remain explicitly marked as pending/deferred where applicable.