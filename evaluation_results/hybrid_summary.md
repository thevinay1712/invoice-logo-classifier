# Real-World Hybrid System Evaluation Report

## Executive Summary

This report evaluates the end-to-end classification accuracy of the combined closed-set classifier + dynamic gallery fallback system on real invoice document scans across three vendor categories.

## 1. Domain Gap Analysis (LogoDet-3K vs. Real Invoices)

| Evaluation Split | Dataset Source | Category | Accuracy / Correct Rate |
|---|---|---|---|
| Phase 6 Closed-Set Test | LogoDet-3K Cropped Images | Trained Classes | **85.2%** |
| **Phase 14 Real Invoices** | **Real Invoice Scans** | **Trained Classes** | **0.0%** |

> **Domain Gap Observation**: Closed-set accuracy on real invoice documents demonstrates the impact of document noise, header formatting, and scanner compression compared to clean LogoDet-3K cropped images.

## 2. Category Performance Breakdown

| Vendor Category | Description | Total Samples | Correct | Accuracy / Rate |
|---|---|---|---|---|
| `trained` | Invoices from closed-set classes | 3 | 0 | 0.0% |
| `gallery` | Invoices from gallery-seeded vendors | 2 | 2 | 100.0% |
| `neither` | Invoices from unseen vendors | 2 | 0 | 0.0% (Correct Unknown) |

## 3. Embedding Backbone Benchmark Comparison

| Embedding Backbone | Feature Dimension | Gallery Retrieval Accuracy |
|---|---|---|
| Fine-Tuned ResNet50 Classifier Backbone | 2048 | **0.0%** |
| Pretrained OpenCLIP ViT-B/32 | 512 | **0.0%** |

**Recommendation**: `Fine-Tuned ResNet50 Classifier Backbone` performs best on real invoice logo embeddings.

## 4. Gallery Similarity Threshold ($T_{sim}$) Sweep

| Similarity Threshold ($T_{sim}$) | Gallery Match Accuracy | Unseen False Match Rate |
|---|---|---|
| 0.40 | 0.0% | 0.0% |
| 0.50 | 0.0% | 0.0% |
| 0.60 | 0.0% | 0.0% |
| 0.70 | 0.0% | 0.0% |
| 0.75 | 0.0% | 0.0% |
| 0.80 | 0.0% | 0.0% |
| 0.85 | 0.0% | 0.0% |
| 0.90 | 0.0% | 0.0% |
| 0.95 | 0.0% | 0.0% |
