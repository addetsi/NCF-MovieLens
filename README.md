# Neural Collaborative Filtering (NCF) for Movie Recommendations

## Overview
This project implements a Neural Collaborative Filtering (NCF) model to predict user-movie interactions using the MovieLens 1M dataset. The model combines matrix factorization with deep learning to provide personalized recommendations. The implementation includes data preprocessing, model training, hyperparameter tuning, and evaluation using Recall@10 and NDCG@10 metrics.

## Features
- Neural network-based collaborative filtering implementation
- User and movie embedding generation
- Negative sampling strategies for improved training
- Hyperparameter optimization for model architecture
- Evaluation using industry-standard metrics (Recall@10, NDCG@10)
- Early stopping mechanism to prevent overfitting

## Requirements
- Python 3.8+
- PyTorch 2.6.0
- scikit-learn 1.6.1
- numpy 2.0.2
- pandas 2.1.0

## Dataset
Download the MovieLens 1M dataset from the [official website](https://grouplens.org/datasets/movielens/1m/) and place the `ratings.dat` and `movies.dat` files in the `data/ml-1m/` directory.

## Usage
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Run NCF.py File**:
   ```bash
   python NCF.py
   ```

## Results
- Recall@10 and NDCG@10 metrics of approximately 0.7 and 0.8 respectively for various hyperparameter configurations
- Best performance achieved with embedding dimension = 64, MLP layers = [128, 64, 32], and negative sample number = 4

## Roadmap
- **Interactive Visualization Dashboard** (In Progress): Building an interactive dashboard for visualization 

