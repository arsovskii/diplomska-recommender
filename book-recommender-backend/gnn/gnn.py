# import torch
# from torch import tensor, Tensor
# from torch_geometric.data import HeteroData
# from torch_geometric.nn import GATConv, to_hetero
# from torch_geometric.loader import LinkNeighborLoader
# import torch_geometric.transforms as T
# import torch.nn as nn
# import pandas as pd
# import pickle
# from tqdm import tqdm
# from typing import List, Dict, Tuple
# import os

# # Constants
# HIDDEN_CHANNELS = 64
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# TARGET_EDGE = ("user", "review", "book")
# REV_TARGET_EDGE = ("book", "rev_review", "user")

# # Load static data
# try:
#     result_book_features = pd.read_csv("./gnn/graphdata/book_features.csv")
#     with open("./gnn/graphdata/saved_dictionary.pkl", "rb") as f:
#         user_id_to_node_id = pickle.load(f)
#     review_edges = torch.load("./gnn/graphdata/review_edges.pt").to(DEVICE)
#     model_state = torch.load("./gnn/graphdata/gat_model_3.pth")
# except Exception as e:
#     print(f"Error loading data: {e}")
#     raise

# class GAT(torch.nn.Module):
#     def __init__(self, hidden_channels: int, out_channels: int):
#         super().__init__()
#         self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
#         self.conv2 = GATConv(hidden_channels, out_channels, add_self_loops=False)

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index).relu()
#         return self.conv2(x, edge_index)

# class Classifier(torch.nn.Module):
#     def forward(self, x_user: Tensor, x_book: Tensor, edge_label_index: Tensor) -> Tensor:
#         edge_feat_user = x_user[edge_label_index[0]].to(dtype=x_book.dtype)
#         edge_feat_book = x_book[edge_label_index[1]]
#         return (edge_feat_user * edge_feat_book).sum(dim=-1)

# class GATModel(torch.nn.Module):
#     def __init__(self, hidden_channels: int, num_users: int, num_books: int):
#         super().__init__()
#         self.user_emb = torch.nn.Embedding(num_users, hidden_channels)
#         self.book_emb = torch.nn.Embedding(num_books, hidden_channels)
#         self.gnn = GAT(hidden_channels, hidden_channels)
#         self.classifier = Classifier()

#     def forward(self, data, edge_index):
#         x_dict = {
#             "user": self.user_emb(data["user"].node_id),
#             "book": self.book_emb(data["book"].node_id),
#         }
#         x_dict = self.gnn(x_dict, data.edge_index_dict)
#         return self.classifier(x_dict["user"], x_dict["book"], edge_index)

# # Initialize model globally
# model = GATModel(
#     hidden_channels=HIDDEN_CHANNELS,
#     num_users=len(user_id_to_node_id) + 10,
#     num_books=len(result_book_features)
# ).to(DEVICE)
# model.load_state_dict(model_state["model_state_dict"])

# def create_graph_data(edge_index_user: Tensor = None) -> HeteroData:
#     """Create a HeteroData graph with optional user edges."""
#     data = HeteroData()

#     # Set node IDs and features
#     data["book"].node_id = torch.arange(len(result_book_features)).to(DEVICE)
#     data["user"].node_id = torch.arange(len(user_id_to_node_id) + 10).to(DEVICE)
#     data["book"].x = tensor(result_book_features.values, dtype=torch.float).to(DEVICE)

#     # Set edges
#     if edge_index_user is not None:
#         data["user", "review", "book"].edge_index = torch.cat(
#             [review_edges, edge_index_user], dim=1
#         )
#     else:
#         data["user", "review", "book"].edge_index = review_edges

#     # Make graph undirected
#     data = T.ToUndirected()(data)
#     data.validate()
#     return data

# def train_model(data: HeteroData, edge_index_user: Tensor, epochs: int = 5):
#     """Train the model with new user ratings."""
#     optimizer = torch.optim.Adam(model.parameters())
#     loss_fn = torch.nn.BCELoss()

#     loader = LinkNeighborLoader(
#         data=data,
#         num_neighbors=[-1, -1],
#         neg_sampling_ratio=2.0,
#         edge_label_index=(TARGET_EDGE, edge_index_user),
#         batch_size=500,
#         shuffle=True,
#     )

#     model.train()
#     for epoch in range(epochs):
#         total_loss = total_examples = 0
#         for batch in loader:
#             batch = batch.to(DEVICE)
#             optimizer.zero_grad()

#             pred = torch.sigmoid(model(batch, batch[TARGET_EDGE].edge_label_index))
#             loss = loss_fn(pred, batch[TARGET_EDGE].edge_label)

#             loss.backward()
#             optimizer.step()

#             total_loss += float(loss) * pred.numel()
#             total_examples += pred.numel()

#         print(f"Epoch: {epoch+1:03d}, Loss: {total_loss / total_examples:.4f}")

# def get_predictions(user_node_id: int) -> List[Tuple[int, float]]:
#     """Get predictions for all books for a given user."""
#     model.eval()

#     # Create prediction graph
#     pred_data = HeteroData()
#     book_nodes = torch.arange(len(result_book_features), dtype=torch.long).to(DEVICE)
#     user_nodes = torch.full((len(book_nodes),), user_node_id, dtype=torch.long).to(DEVICE)
#     edge_index = torch.stack([user_nodes, book_nodes], dim=0)

#     pred_data["book"].node_id = torch.arange(len(result_book_features)).to(DEVICE)
#     pred_data["user"].node_id = torch.arange(len(user_id_to_node_id) + 10).to(DEVICE)
#     pred_data["book"].x = tensor(result_book_features.values, dtype=torch.float).to(DEVICE)
#     pred_data["user", "review", "book"].edge_index = edge_index
#     pred_data["user", "review", "book"].edge_label_index = torch.ones(edge_index.size(-1), dtype=torch.long).to(DEVICE)

#     pred_data = T.ToUndirected()(pred_data)
#     pred_data = pred_data.to(DEVICE)

#     predictions = []
#     loader = LinkNeighborLoader(
#         data=pred_data,
#         num_neighbors=[20, 10],
#         batch_size=384,
#         shuffle=False,
#         edge_index=(TARGET_EDGE, edge_index),
#         edge_label_index=(TARGET_EDGE, pred_data[TARGET_EDGE].edge_label_index),
#     )

#     with torch.no_grad():
#         for batch in loader:
#             batch = batch.to(DEVICE)
#             out = model(batch, batch[TARGET_EDGE].edge_label_index)
#             probabilities = torch.sigmoid(out)
#             predictions.append(probabilities.item())

#     # Save predictions for debugging if needed
#     with open("predictions.pkl", "wb") as file:
#         pickle.dump(predictions, file)

#     return list(enumerate(predictions))

# def train_with_user_ratings(ratings: List[int]) -> List[Tuple[int, float]]:
#     """Main function to process new ratings and return recommendations."""
#     try:
#         # Assign new user ID
#         user_node_id = len(user_id_to_node_id)

#         # Create edge index for the new ratings
#         book_nodes = torch.tensor(ratings, dtype=torch.long).to(DEVICE)
#         user_nodes = torch.full((len(ratings),), user_node_id, dtype=torch.long).to(DEVICE)
#         edge_index_user = torch.stack([user_nodes, book_nodes], dim=0)

#         # Create training graph with new ratings
#         train_data = create_graph_data(edge_index_user)

#         # Train model with new ratings
#         train_model(train_data, edge_index_user)

#         # Get predictions for all books
#         predictions = get_predictions(user_node_id)

#         # Sort predictions by probability
#         predictions.sort(key=lambda x: x[1], reverse=True)

#         return predictions

#     except Exception as e:
#         print(f"Error in parse_ratings: {e}")
#         return []

from typing import List
import torch
from torch import tensor, Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, to_hetero
from torch_geometric.loader import LinkNeighborLoader
import torch_geometric.transforms as T
import torch.nn as nn

from tqdm import tqdm

import pandas as pd
import pickle
import os

os.environ["TORCH"] = torch.__version__
print(torch.__version__)
# torch-2.4.0+cu121
target_edge = ("user", "review", "book")
rev_target_edge = ("book", "rev_review", "user")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model_state = torch.load("./gnn/graphdata/gat_model_3.pth")
review_edges = torch.load("./gnn/graphdata/review_edges.pt")
review_edges = review_edges.to(device)

user_id_to_node_id = {}

with open("./gnn/graphdata/saved_dictionary.pkl", "rb") as f:
    user_id_to_node_id = pickle.load(f)

result_book_features = pd.read_csv("./gnn/graphdata/book_features.csv")


data = HeteroData()


data["book"].node_id = torch.arange(len(result_book_features))
data["user"].node_id = torch.arange(len(user_id_to_node_id) + 10)

data["book"].x = tensor(result_book_features.values, dtype=torch.float)

data["user", "review", "book"].edge_index = review_edges

data = T.ToUndirected()(data)

assert data["user"].num_nodes == len(user_id_to_node_id) + 10
assert data["book"].num_features == 23
assert data["user"].num_features == 0

data.validate()


class GAT_3(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.conv2 = GATConv(hidden_channels, out_channels, add_self_loops=False)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)

        return x


class Classifier(torch.nn.Module):
    def forward(
        self, x_user: Tensor, x_book: Tensor, edge_label_index: Tensor
    ) -> Tensor:
        # Converting node embeddings to edge-level representations:

        edge_feat_user = x_user[edge_label_index[0]].to(dtype=x_book.dtype)
        edge_feat_book = x_book[edge_label_index[1]]

        # Applying dot-product to get a prediction per supervision edge:
        return (edge_feat_user * edge_feat_book).sum(dim=-1)


class GATModel_3(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.user_emb = torch.nn.Embedding(data["user"].num_nodes, hidden_channels)
        self.book_emb = torch.nn.Embedding(data["book"].num_nodes, hidden_channels)

        self.gnn = GAT_3(hidden_channels, hidden_channels)
        self.gnn = to_hetero(self.gnn, data.metadata())
        self.classifier = Classifier()

    def forward(self, data, edge_index):
        x_dict = {
            "user": self.user_emb(data["user"].node_id),
            "book": self.book_emb(data["book"].node_id),
        }

        x_dict = self.gnn(x_dict, data.edge_index_dict)

        pred = self.classifier(
            x_dict["user"],
            x_dict["book"],
            edge_index,
        )

        return pred


gatModel_3 = GATModel_3(hidden_channels=64).to(
    device
)  # Replace with your model class and initialization
optimizer = torch.optim.Adam(
    gatModel_3.parameters(),
    lr=0.00001,
)  # Replace with the optimizer you used

# Load the saved state dicts into the model and optimizer
gatModel_3.load_state_dict(model_state["model_state_dict"])
optimizer.load_state_dict(model_state["optimizer_state_dict"])


def train_with_user_ratings(user_ratings):
    gatModel_3.load_state_dict(model_state["model_state_dict"])
    optimizer.load_state_dict(model_state["optimizer_state_dict"])
    print(user_ratings)

    print(f"Training with {len(user_ratings)} user ratings")

    user_node_id = len(user_id_to_node_id)
    new_book_node_ids = torch.tensor(user_ratings, dtype=torch.long).to(device)
    user_ratings_tensor = torch.full(
        (len(new_book_node_ids),), user_node_id, dtype=torch.long
    ).to(device)
    edge_index_user = torch.stack([user_ratings_tensor, new_book_node_ids], dim=0)

    old_edge_index = data["user", "review", "book"].edge_index

    # Create training graph
    train_data = HeteroData()
    train_data["book"].node_id = torch.arange(len(result_book_features))
    train_data["user"].node_id = torch.arange(len(user_id_to_node_id) + 10)
    train_data["book"].x = tensor(result_book_features.values, dtype=torch.float)

    train_data["user", "review", "book"].edge_index = torch.cat(
        [data["user", "review", "book"].edge_index, edge_index_user], dim=1
    )

    train_data = T.ToUndirected()(train_data)
    train_data.validate()

    assert train_data["user"].num_nodes == len(user_id_to_node_id) + 10
    assert train_data["book"].num_features == 23
    assert train_data["user"].num_features == 0

    # Batching our dataset to prevent out of memory errors
    data_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=[-1, -1],
        neg_sampling_ratio=4.0,  # Increased negative sampling
        edge_label_index=(("user", "review", "book"), edge_index_user),
        batch_size=256,  # Adjusted batch size
        shuffle=True,
    )

    pos_weight = torch.tensor([4.0]).to(device)  # Adjust based on your data
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    gatModel_3.train()

    for epoch in range(5):
        total_loss = total_examples = 0
        for batch in tqdm(data_loader):
            optimizer.zero_grad()
            
            batch = batch.to(device)
            pred = gatModel_3(batch, batch["user", "review", "book"].edge_label_index)
            
            # Apply label smoothing
            ground_truth = batch[("user", "review", "book")].edge_label
            smoothing = 0.1
            ground_truth = ground_truth * (1 - smoothing) + smoothing / 2
            
            loss = loss_fn(pred, ground_truth)
            
            # Add L2 regularization
            l2_lambda = 0.01
            l2_reg = torch.tensor(0.).to(device)
            for param in gatModel_3.parameters():
                l2_reg += torch.norm(param)
            loss += l2_lambda * l2_reg
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(gatModel_3.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()
        
        print(f"Epoch: {epoch+1:03d}, Loss: {total_loss / total_examples:.4f}")

    return get_predictions(user_node_id, user_ratings)
   

    # preds = []
    # ground_truths = []
    # edges = []

    # gatModel_3.eval()

    # for sampled_data in tqdm(val_loader):
    #     with torch.no_grad():  # we set this to not calculate gradients
    #         edges.append(sampled_data.edge_index_dict[target_edge])
    #         sampled_data.to(device)
    #         out = gatModel_3(sampled_data)
    #         p = torch.sigmoid(gatModel_3(sampled_data))
    #         preds.append(p)
    #         ground_truths.append(sampled_data[target_edge].edge_label)
    # print("test")


def get_predictions(user_node_id: int, userratings: List[int]):
    """Get predictions for all books for a given user."""
    gatModel_3.eval()
    
    # Create prediction data
    pred_data = HeteroData()
    
    # Get all book IDs except those the user has already rated
    all_book_ids = set(range(len(result_book_features)))
    rated_books = set(userratings)
    books_to_predict = list(all_book_ids - rated_books)
    
    # Set up nodes
    pred_data["book"].node_id = torch.arange(len(result_book_features)).to(device)
    pred_data["user"].node_id = torch.arange(len(user_id_to_node_id) + 10).to(device)
    pred_data["book"].x = tensor(result_book_features.values, dtype=torch.float).to(device)
    
    # Include existing user ratings in the graph for better context
    edge_list = []
    if userratings:
        existing_user_nodes = torch.full((len(userratings),), user_node_id, dtype=torch.long).to(device)
        existing_book_nodes = torch.tensor(userratings, dtype=torch.long).to(device)
        existing_edges = torch.stack([existing_user_nodes, existing_book_nodes], dim=0)
        edge_list.append(existing_edges)
    
    # Add all existing edges from the training data to maintain graph structure
    edge_list.append(data["user", "review", "book"].edge_index.to(device))
    
    # Combine all edges
    pred_data["user", "review", "book"].edge_index = torch.cat(edge_list, dim=1)
    
    # Make the graph undirected to allow information flow
    pred_data = T.ToUndirected()(pred_data)
    pred_data = pred_data.to(device)
    
    # Process in smaller batches
    batch_size = 512
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(books_to_predict), batch_size):
            batch_books = books_to_predict[i:i + batch_size]
            batch_users = torch.full((len(batch_books),), user_node_id, dtype=torch.long).to(device)
            batch_books_tensor = torch.tensor(batch_books, dtype=torch.long).to(device)
            batch_edge_index = torch.stack([batch_users, batch_books_tensor], dim=0)
            
            # Get predictions
            out = gatModel_3(pred_data, batch_edge_index)
            
            # Apply softmax with temperature scaling
            temperature = 1.5
            scaled_logits = out / temperature
            probabilities = torch.sigmoid(scaled_logits)
            
            # Store predictions with book features for diversity calculation
            for book_id, prob in zip(batch_books, probabilities.cpu().numpy()):
                book_features = result_book_features.iloc[book_id]
                predictions.append({
                    'book_id': book_id,
                    'probability': float(prob),
                    'features': book_features
                })
    
    # Sort by probability
    predictions.sort(key=lambda x: x['probability'], reverse=True)
    
    # Apply diversity filtering
    diverse_predictions = []
    feature_memory = []
    similarity_threshold = 0.9
    
    def compute_feature_similarity(features1, features2):
        return float(torch.cosine_similarity(
            torch.tensor(features1.values),
            torch.tensor(features2.values),
            dim=0
        ))
    
    for pred in predictions:
        # Skip very low probability predictions
        if pred['probability'] < 0.5:
            continue
            
        # Check similarity with already selected books
        is_diverse = True
        for selected_features in feature_memory:
            similarity = compute_feature_similarity(pred['features'], selected_features)
            if similarity > similarity_threshold:
                is_diverse = False
                break
        
        if is_diverse:
            diverse_predictions.append({pred['book_id']: pred['probability']})
            feature_memory.append(pred['features'])
            
            # Limit number of recommendations
            if len(diverse_predictions) >= 10:
                break

    if len(diverse_predictions) < 10:
        for pred in predictions:
            if len(diverse_predictions) >= 10:
                break
            if pred['book_id'] not in [list(x.keys())[0] for x in diverse_predictions]:
                diverse_predictions.append({pred['book_id']: pred['probability']})

    print(diverse_predictions)
    return predictions, diverse_predictions
    # gatModel_3.eval()

    # predictions = []

    # with torch.no_grad():
    #     for book_id in tqdm(range(len(result_book_features))):
    #         # Create a single prediction instance
    #         if book_id in userratings:
    #             continue
    #         pred_data = HeteroData()

    #         # Set up nodes
    #         pred_data["book"].node_id = torch.tensor([book_id]).to(device)
    #         pred_data["user"].node_id = torch.tensor([user_node_id]).to(device)
    #         pred_data["book"].x = (
    #             tensor(result_book_features.iloc[book_id].values, dtype=torch.float)
    #             .unsqueeze(0)
    #             .to(device)
    #         )

    #         # Set up edge
    #         edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(
    #             device
    #         )  # Single edge connecting user to book
    #         pred_data["user", "review", "book"].edge_index = edge_index

    #         pred_data = T.ToUndirected()(pred_data)
    #         pred_data = pred_data.to(device)

    #         # Get prediction
    #         out = gatModel_3(pred_data, pred_data[target_edge].edge_index)
    #         probability = torch.sigmoid(out).item()
    #         predictions.append({book_id: probability})
    # # print(predictions)
    # print(sorted(predictions, key=lambda x: list(x.values())[0], reverse=False)[:])
    # # Save predictions for debugging if needed
    # # with open("predictions2.pkl", "wb") as file:
    # #     pickle.dump(predictions, file)

    # return list(enumerate(predictions))


# def predict_for_user(user_id):
#     # Get the user node ID for the specific user.
#     # Ensure the user_id exists in user_id_to_node_id or handle it accordingly.
#     user_node_id = user_id_to_node_id.get(user_id, len(user_id_to_node_id))
#     print(user_node_id)
#     user_node_id = 3
#     # Create edges between the user and all books for prediction.
#     all_book_node_ids = torch.arange(len(result_book_features), dtype=torch.long).to(
#         device
#     )
#     user_node_ids = torch.full(
#         (len(all_book_node_ids),), user_node_id, dtype=torch.long
#     ).to(device)

#     # Create edge index for the specific user interacting with every book.
#     edge_index_user = torch.stack([user_node_ids, all_book_node_ids], dim=0)

#     # Update the graph data with these edges (for prediction purposes only).
#     newdata = HeteroData()
#     newdata["book"].node_id = torch.arange(
#         len(result_book_features), dtype=torch.long
#     ).to(device)

#     newdata["user"].node_id = torch.arange(
#         len(user_id_to_node_id) + 10, dtype=torch.long
#     ).to(device)

#     newdata["book"].x = tensor(result_book_features.values, dtype=torch.float).to(
#         device
#     )
#     print(edge_index_user)
#     newdata["user", "review", "book"].edge_index = edge_index_user
#     print(edge_index_user.size(-1))
#     newdata["user", "review", "book"].edge_label_index = torch.tensor(
#         torch.ones(edge_index_user.size(-1), dtype=torch.long)
#     ).to(device)
#     print(newdata["user", "review", "book"].edge_label_index)
#     edge_index = edge_index_user


#     # Ensure data is on the correct device.

#     newdata = T.ToUndirected()(newdata)
#     assert newdata["user"].num_nodes == len(user_id_to_node_id) + 10
#     assert newdata["book"].num_features == 23
#     assert newdata["user"].num_features == 0

#     newdata = newdata.to(device)
#     print(newdata)
#     transform = T.RandomLinkSplit(
#         num_val=0.0,
#         num_test=1,
#         disjoint_train_ratio=0,  # supervision ratio
#         neg_sampling_ratio=2.0,
#         add_negative_train_samples=False,
#         edge_types=[target_edge],
#         rev_edge_types=[rev_target_edge],
#     )
#     train_data_all, val_data_all, test_data_all = transform(newdata)
#     print("Training data:")
#     print("==============")
#     print(train_data_all)
#     print()
#     print("Validation data:")
#     print("================")
#     print(val_data_all)
#     print("Test data:")
#     print("================")
#     print(test_data_all)
#     print(newdata)
#     preds = []

#     edges = []

#     # Make predictions using the trained model.
#     gatModel_3.eval()  # Set the model to evaluation mode.
#     # Defining the validation seed edges:
#     print("|")
#     print(edge_index_user)
#     print(edge_index)
#     print("|")
#     # print(newdata["user", "review", "book"].edge_index)
#     data_loader = LinkNeighborLoader(
#         data=newdata,
#         num_neighbors=[20, 10],
#         batch_size=3 * 128,
#         shuffle=False,
#         edge_index=(("user", "review", "book"), edge_index),
#         edge_label_index=(
#             ("user", "review", "book"),
#             newdata["user", "review", "book"].edge_label_index,
#         ),
#     )
#     for sampled_data in tqdm(data_loader):

#         with torch.no_grad():
#             # Perform a forward pass with the model.

#             out = gatModel_3(
#                 sampled_data,
#                 edge_index=sampled_data["user", "review", "book"].edge_label_index,
#             )
#             # print(out[0])
#             probabilities = torch.sigmoid(out[0])
#             preds.append(probabilities.item())
#             edges.append(sampled_data.edge_index_dict[("user", "review", "book")])

#             if len(edges) < 3:
#                 print("pod")
#                 print(sampled_data["user", "review", "book"].edge_index)
#                 print("mejdzu")
#                 print(sampled_data["user", "review", "book"].edge_label_index)
#                 print("nad")
#                 # print(edges)
#                 # print(sampled_data)
#                 print()

#             # edges.append(sampled_data.edge_index_dict[("user", "review", "book")])
#             # sampled_data.to(device)
#             # out = gatModel_3(sampled_data)
#             # p = torch.sigmoid(gatModel_3(sampled_data))
#             # preds.append(p)
#             # ground_truths.append(sampled_data[("user", "review", "book")].edge_label)
#             # print("ovde?")
#             # print(p)
#             # pred = gatModel_3(newdata, edge_index_user)

#             # pred = torch.sigmoid(pred)  # Convert logits to probabilities.
#             # print("ovde?")
#             # print(pred)

#     print(preds)
#     with open("predictions2.pkl", "wb") as file:
#         # A new file will be created
#         pickle.dump(preds, file)
#     with open("edges2.pkl", "wb") as file:
#         # A new file will be created
#         pickle.dump(edges, file)
#     # `pred` now contains the probabilities of interaction for the user with each book.
#     # Convert predictions to a list with book indices and their respective probabilities.
#     predictions = list(zip(all_book_node_ids.cpu().numpy(), pred.cpu().numpy()))

#     # Sort predictions by probability in descending order for top recommendations.
#     predictions.sort(key=lambda x: x[1], reverse=True)

#     # Print or return the top recommendations.
#     print(f"Top recommendations for user {user_id}:")
#     for book_id, probability in predictions[:100]:
#         print(f"Book ID: {book_id}, Probability: {probability:.4f}")

#     return predictions
