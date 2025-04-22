import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict

ratings_path = "data\\ml-1m\\ratings.dat" #path to the ratings
movies_path = "data\\ml-1m\\movies.dat" # path to the movies
all_movies = set(range(0, 3952))  #0-based indexing set for movies

def get_true_labels(negative_sample_number):
    """Get true labels and track watched movies per user."""
    true_labels = []
    user_watched = defaultdict(set)  # Track watched movies per user
    with open(ratings_path, 'r') as data_file:
        for line in data_file:
            parts = line.strip().split('::') # extract user, movie and rating
            user_id = int(parts[0]) - 1  # Convert to 0-based indexing
            movie_id = int(parts[1]) - 1  # Convert to 0-based indexing
            rating = 1 if int(parts[2]) >= 4 else 0 # convert to binary rating
            true_labels.append((user_id, movie_id, rating))
            user_watched[user_id].add(movie_id)
    
    # Negative sampling
    for user_id, watched_movies in user_watched.items():
        unwatched_movies = list(set(range(3952)) - watched_movies)  # 3952 movies, 0-based
        for _ in range(min(negative_sample_number, len(unwatched_movies))):
            neg_movie = random.choice(unwatched_movies)
            true_labels.append((user_id, neg_movie, 0))
    
    # Return  true_labels
    return true_labels

def split_data(true_labels):
    """Split data into train, test, and validation sets."""
    train_data, test_val_data = train_test_split(true_labels, train_size=0.7, random_state=42)
    test_data, val_data = train_test_split(test_val_data, train_size=0.5, random_state=42)
    return train_data, test_data, val_data

#NCF model
class NCF(nn.Module):
    def __init__(self, n_users, n_movies, embedding_dim, mlp_layers, dropout=0.2):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding = nn.Embedding(n_movies, embedding_dim)
        
        # MLP branch with configurable layers and dropout
        layers = []
        in_features = 2 * embedding_dim  # Input is concatenation of user and movie embeddings
        for out_features in mlp_layers:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))  # Added dropout to prevent overfitting
            in_features = out_features
        self.mlp = nn.Sequential(*layers)
        
        # Fusion layer (GMF + MLP)
        self.fusion_layer = nn.Linear(embedding_dim + mlp_layers[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_ids, movie_ids):
        user_vec = self.user_embedding(user_ids)
        movie_vec = self.movie_embedding(movie_ids)
        
        # GMF branch: element-wise product
        gmf_output = torch.mul(user_vec, movie_vec)
        
        # MLP branch: concatenate embeddings
        mlp_input = torch.cat([user_vec, movie_vec], dim=1)
        mlp_output = self.mlp(mlp_input)
        
        # Combine GMF and MLP
        combined = torch.cat([gmf_output, mlp_output], dim=1)
        output = self.fusion_layer(combined)
        return self.sigmoid(output)

def get_data_loader(data, batch_size=256, shuffle=True):
    """Convert data to PyTorch DataLoader."""
    userIds = torch.tensor([x[0] for x in data], dtype=torch.long)
    movieIds = torch.tensor([x[1] for x in data], dtype=torch.long)
    ratings = torch.tensor([x[2] for x in data], dtype=torch.float32)
    dataset = TensorDataset(userIds, movieIds, ratings)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def predict_ranking(model,user_id,rated_movies,device): # predict the ranking of rated_movies for user_id
    if not rated_movies:
        return [] # return an empty list if user did not rate any movies
    user_ids=torch.tensor([user_id]*len(rated_movies),dtype=torch.long).to(device)
    movie_ids=torch.tensor(list(rated_movies.keys()),dtype=torch.long).to(device)
    with torch.no_grad():
        predictions=model(user_ids, movie_ids).flatten().cpu().numpy() # predict ratings
    ratings = [
        (user_id, movie_id, float(pred)) 
        for movie_id, pred in zip(rated_movies.keys(), predictions) # create list of tuples with (user_id,movie_id,rating)
    ]
    return sorted(ratings,key=lambda x: x[2],reverse=True) # sort the list of ratings

def recall_and_NDCG_at_k(predicted_ranking,rated_movies,k=10): # calculate the recall and NDCG
    relevant_top_k=0
    DCG_at_k=0
    IDCG_at_k=0
    relevant_total=sum(1 for positive in rated_movies.values() if positive==1) # calculate total amount of relevant movies
    for i in range(k):
        if len(predicted_ranking)==i:
            break
        if rated_movies[predicted_ranking[i][1]]==1: # if this movie is relevant
            DCG_at_k+=1/np.log2(i+2) # update DCG
            relevant_top_k+=1 # increase amount of relevant movies in top k
        if relevant_total>i: # ensures the ideal order contains either all relevent movies, or k if there are more than k
            IDCG_at_k+=1/np.log2(i+2) # update IDCG
    if IDCG_at_k==0:
        raise Exception("NDCG cannot be calculated, because IDCG is 0")
    if relevant_total==0:
        raise Exception("Recall cannot be calculated, because relevant_total is 0")
    return relevant_top_k/relevant_total,DCG_at_k/IDCG_at_k # calculate recall and NDCG

def evaluation(model,test_data,device):
    skipped_users=0
    print("Starting evaluation")
    rated_movies=[]
    for user in range(6040): # for each user, make a dictionary, with movie id as key and rating as value
        rated_movies.append({movie[1]:movie[2] for movie in test_data if movie[0]==user})
    recalls_at_10=[]
    NDCGs_at_10=[]
    for user in range(6040): # for each user
        predicted_ranking=predict_ranking(model,user,rated_movies[user],device) # predict ranking
        try:
            recall_at_10,NDCG_at_10=recall_and_NDCG_at_k(predicted_ranking,rated_movies[user]) # calculate recall and NDCG
            recalls_at_10.append(recall_at_10)
            NDCGs_at_10.append(NDCG_at_10)
        except:
            skipped_users+=1
    print(f"Recall and/or NDCG could not be calculated for {skipped_users} users.")
    length=len(recalls_at_10)
    average_recall_at_10=sum(recalls_at_10)/length # calculate average recall
    print(f"Average recall@10: {average_recall_at_10}")
    average_NDCG_at_10=sum(NDCGs_at_10)/length # calculate average NDCG
    print(f"Average NDCG@10: {average_NDCG_at_10}")
    print("Finished evaluation")

def train_model(model, train_loader, val_loader, device, epochs=10, lr=0.001, patience=3):
    """Training loop with early stopping."""
    optimizer = optim.Adam(model.parameters(), lr=lr)  
    criterion = nn.BCELoss()
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for userIds, movieIds, ratings in train_loader:
            userIds, movieIds, ratings = userIds.to(device), movieIds.to(device), ratings.to(device)
            optimizer.zero_grad()
            outputs = model(userIds, movieIds).squeeze()
            loss = criterion(outputs, ratings)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * userIds.size(0)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for userIds, movieIds, ratings in val_loader:
                userIds, movieIds, ratings = userIds.to(device), movieIds.to(device), ratings.to(device)
                outputs = model(userIds, movieIds).squeeze()
                val_loss += criterion(outputs, ratings).item() * userIds.size(0)
        
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    model.load_state_dict(torch.load('best_model.pth'))
    return model

def main():
    # Define hyperparameter configurations to test
    configs = [
        {"embedding_dim": 16, "mlp_layers": [32, 16, 8], "negative_sample_number": 1},
        {"embedding_dim": 32, "mlp_layers": [64, 32, 16], "negative_sample_number": 1},
        {"embedding_dim": 64, "mlp_layers": [128, 64, 32], "negative_sample_number": 1},
        {"embedding_dim": 64, "mlp_layers": [64, 32, 16], "negative_sample_number": 1},
        {"embedding_dim": 16, "mlp_layers": [32, 16, 8], "negative_sample_number": 2},
        {"embedding_dim": 32, "mlp_layers": [64, 32, 16], "negative_sample_number": 2},
        {"embedding_dim": 64, "mlp_layers": [128, 64, 32], "negative_sample_number": 2},
        {"embedding_dim": 64, "mlp_layers": [64, 32, 16], "negative_sample_number": 2},
        {"embedding_dim": 16, "mlp_layers": [32, 16, 8], "negative_sample_number": 4},
        {"embedding_dim": 32, "mlp_layers": [64, 32, 16], "negative_sample_number": 4},
        {"embedding_dim": 64, "mlp_layers": [128, 64, 32], "negative_sample_number": 4},
        {"embedding_dim": 64, "mlp_layers": [64, 32, 16], "negative_sample_number": 4},
    ]

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loop over configurations to test different hyperparameters
    for config in configs:
        # Get and split data
        true_labels = get_true_labels(config['negative_sample_number'])
        train_data, test_data, val_data = split_data(true_labels)

        # Create data loaders
        train_loader = get_data_loader(train_data, batch_size=256)
        val_loader = get_data_loader(val_data, batch_size=256, shuffle=False)

        # Get number of unique users and movies
        num_users = max([x[0] for x in true_labels]) + 1
        num_movies = 3952

        print(f"\nTraining with embedding_dim={config['embedding_dim']}, mlp_layers={config['mlp_layers']}, negative_sample_number={config['negative_sample_number']}")
        
        # Create and train model with current configuration
        model = NCF(num_users, num_movies, config["embedding_dim"], config["mlp_layers"]).to(device)
        model = train_model(model, train_loader, val_loader, device)
        
        # Evaluate on test set
        evaluation(model,test_data,device)

if __name__ == "__main__":
    main()