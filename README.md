# LLaMA 3 QLoRA Fine-tuning for Sentiment Analysis 🦙🎭

This repository contains an adapted script for fine-tuning the **LLaMA 3 (8B)** model using **QLoRA (Quantized Low-Rank Adaptation)** for sentiment analysis and emotion classification in text.

🏆 **Background:** This code is an QLoRA adaptation of the approach I used in my participation for the **[EmoSPeech 2024 Task - Multimodal Speech-text Emotion Recognition in Spanish](https://codalab.lisn.upsaclay.fr/competitions/17647#results)**, where I achieved top places on the leaderboard.

## 🚀 Overview
The script is highly optimized using **QLoRA** (4-bit quantization) and specific data type patches to train massive models on resource-constrained GPUs. It is designed to be executed remotely via the official Kaggle API.

## 🛠️ Usage (Kaggle API)

1. Ensure you have the Kaggle CLI installed and your `kaggle.json` configured.
2. Push the code to train remotely on Kaggle's GPUs:
   ```bash
   kaggle kernels push -p .
