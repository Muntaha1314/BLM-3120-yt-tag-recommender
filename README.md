# 🎥 YouTube Tag Recommendation System
**Multi-Label Machine Learning Project**

---

## 📌 Project Overview

This project implements a **YouTube tag recommendation system** using **machine learning**.  
Given a video’s **title and description**, the system predicts the **most relevant tags** that could be assigned to the video.

The goal of this project is to demonstrate:
- Text preprocessing and feature extraction
- Multi-label classification
- Use of multiple machine learning models
- Model comparison and evaluation using appropriate metrics

The project is implemented in **Python** using **scikit-learn**, and all results are displayed directly in the terminal.

---

## 🎯 Objectives

- Build a tag recommender using a **public dataset**
- Apply **multiple machine learning models**
- Compare model performance
- Evaluate predictions using **Precision@K** and **Recall@K**
- Display **real sample predictions** as output

---

## 📂 Dataset

- **Source:** Kaggle (YouTube Trending Videos dataset)
- **File used:** `videos.csv`
- **Features used:**
  - `title`
  - `description`
  - `tags`

### 🔍 Dataset Preprocessing

- Tags are split into lists using `"|"` as a separator
- Only the **top 300 most frequent tags** are kept to reduce sparsity and memory usage
- Videos without valid tags are removed

> Limiting the number of tags is a standard approach in large-scale recommender systems.

---

## 🧠 Machine Learning Models Used

This project implements **multi-label classification** using a **One-Vs-Rest strategy**.

### Models:
1. **Multinomial Naive Bayes**
2. **Decision Tree Classifier**
3. **k-Nearest Neighbors (kNN)**

Each model is trained on the same dataset and evaluated using the same metrics to allow fair comparison.

---

## 🔧 Feature Engineering

- Text features are created by combining:
  - `title + description`
- Text preprocessing includes:
  - Lowercasing
  - Removing URLs and punctuation
  - Stopword removal
- Features are extracted using **TF-IDF Vectorization** (max features: 5000)

---

## 📊 Evaluation Metrics

Because this is a **multi-label problem**, traditional accuracy is not sufficient.

The following metrics are used:

- **Precision@K**: How many of the top-K predicted tags are correct
- **Recall@K**: How many of the true tags are recovered in the top-K predictions

> K = 5

---

## 🖥️ Sample Terminal Output

Each model produces:
- Precision@5
- Recall@5
- A real example showing:
  - Video title
  - True tags
  - Predicted tags

### Example
Model: Naive Bayes
Precision@5: 0.31
Recall@5: 0.22

Sample Prediction
Title: how to cook pasta perfectly
True Tags: ['food', 'cooking']
Predicted Tags: ['food', 'recipe', 'easy']


