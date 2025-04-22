# Neural Collaborative Filtering (NCF) for Movie Recommendations

## Overview
This project implements a Neural Collaborative Filtering (NCF) model to predict user-movie interactions using the MovieLens 1M dataset. The model combines matrix factorization with deep learning to provide personalized recommendations. The implementation includes data preprocessing, model training, hyperparameter tuning, and evaluation using Recall@10 and NDCG@10 metrics.

## Requirements
- Python 3.8+
- PyTorch 2.6.0
- scikit-learn 1.6.1
- numpy 2.0.2

## Dataset
Download the MovieLens 1M dataset from the [official website](https://grouplens.org/datasets/movielens/1m/) and place the `ratings.dat` and `movies.dat` files in the `data/ml-1m/` directory.

## Usage
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
2. **Run NCF.py File**:
    python NCF.py 

## Results:
    Recall@10 and NDCG@10 to be around 0.7 and 0.8 
    for the various hyperparameters.
