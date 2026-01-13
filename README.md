# BERT_Fine_Tunning_on_IMDB 

Sentiment Analysis using Transformers

This repository contains an implementation of BERT fine-tuning for binary sentiment classification on the IMDb movie reviews dataset. The project demonstrates the end-to-end workflow of loading data, preprocessing, fine-tuning a pretrained transformer model, and evaluating performance using standard NLP metrics.

## Project Overview

Task: Binary Sentiment Classification (Positive / Negative)

Model: bert-base-uncased

Frameworks: PyTorch, Hugging Face Transformers & Datasets

Dataset: IMDb Movie Reviews

Type: Fine-tuning a pretrained Large Language Model (LLM)

## Dataset

Source: Hugging Face imdb dataset

Labels:

      0 → Negative

      1 → Positive

### Balanced Subset Used

To reduce training time, a balanced subset was created:

    Split	 Positive	 Negative

    Train	   2000	      2000

    Test	   500	      500

## Model Details

Pretrained Model: bert-base-uncased

Architecture: BertForSequenceClassification

Number of classes: 2

## Training Configuration

Max sequence length: 256

Batch size: 16

Epochs: 8

Optimizer: AdamW

Learning rate: 2e-5

Loss function: CrossEntropyLoss

Hardware: GPU (if available)

## Visualizations & Analysis

The notebook includes:

Class distribution plots (bar & pie charts)

Review length analysis (histogram & boxplot)

Word clouds for positive and negative reviews

Training loss curve

Smoothed loss (moving average)

Confusion matrix for test predictions

## Results

Performance on the balanced test set:

    Accuracy: ~89.0%

    F1 Score: ~88.8%

These results demonstrate effective fine-tuning of BERT for sentiment classification with a relatively small training subset.


