🎥 YouTube Tag Recommendation System

Multi-Label Machine Learning Project

📌 Project Overview

This project implements a YouTube tag recommendation system using machine learning.
Given a video’s title and description, the system predicts the most relevant tags that could be assigned to the video.

The goal of this project is to demonstrate:

Text preprocessing and feature extraction

Multi-label classification

Use of multiple machine learning models

Model comparison and evaluation using appropriate metrics

The project is implemented in Python using scikit-learn, and all results are displayed directly in the terminal.

🎯 Objectives

Build a tag recommender using a public dataset

Apply multiple machine learning models

Compare model performance

Evaluate predictions using Precision@K and Recall@K

Display real sample predictions as output

📂 Dataset

Source: Kaggle (YouTube Trending Videos dataset)

File used: videos.csv

Features used:

title

description

tags

🔍 Dataset Preprocessing

Tags are split into lists using "|" as a separator

Only the top 300 most frequent tags are kept to reduce sparsity and memory usage

Videos without valid tags are removed

Limiting the number of tags is a standard approach in large-scale recommender systems.