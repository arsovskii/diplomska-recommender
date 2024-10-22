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
    gatModel_3.parameters()
)  # Replace with the optimizer you used

# Load the saved state dicts into the model and optimizer
gatModel_3.load_state_dict(model_state["model_state_dict"])
optimizer.load_state_dict(model_state["optimizer_state_dict"])


def train_with_user_ratings(user_ratings):
    print(user_ratings)

    user_node_id = len(user_id_to_node_id)

    new_book_node_ids = torch.tensor(user_ratings, dtype=torch.long).to(device)

    # user_edge_ids = torch.full( new_book_node_ids.size(0), user_node_id, dtype=torch.long).to(device)

    user_ratings_tensor = torch.tensor(
        [user_node_id] * len(new_book_node_ids), dtype=torch.long
    ).to(device)

    

    edge_index_user = torch.stack([user_ratings_tensor, new_book_node_ids], dim=0)

    old_edge_index = data["user", "review", "book"].edge_index

    newdata = HeteroData()

    newdata["book"].node_id = torch.arange(len(result_book_features))
    newdata["user"].node_id = torch.arange(len(user_id_to_node_id) + 10)

    newdata["book"].x = tensor(result_book_features.values, dtype=torch.float)

    newdata["user", "review", "book"].edge_index = torch.cat(
        [old_edge_index, edge_index_user], dim=1
    )

    newdata = T.ToUndirected()(newdata)
    
    assert newdata["user"].num_nodes == len(user_id_to_node_id) + 10
    assert newdata["book"].num_features == 23
    assert newdata["user"].num_features == 0

    newdata.validate()
    
    # Batching our dataset to prevent out of memory errors
    data_loader = LinkNeighborLoader(
        data=newdata,
        num_neighbors=[-1, -1],
        neg_sampling_ratio=2.0,
        edge_label_index=(("user", "review", "book"), edge_index_user),
        batch_size=500,
        shuffle=True,
    )

    loss_fn = torch.nn.BCELoss()

    gatModel_3.train()

    for epoch in range(1, 5):
        total_loss = total_examples = 0
        for sampled_data in tqdm(data_loader):
            optimizer.zero_grad()  # resetting the optimizer gradients

        
            sampled_data.to(device)

            pred = gatModel_3(sampled_data, sampled_data["user", "review", "book"].edge_label_index)
            pred = torch.nn.Sigmoid()(
                pred
            )  # converting the logits to normalized probabilistic output

            ground_truth = sampled_data[
                ("user", "review", "book")
            ].edge_label  # getting the ground truth for the target link
            loss = loss_fn(
                pred, ground_truth
            )  # calculating the loss from the predictions and ground truth

            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()

        print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")

    
    predict_for_user(user_node_id)
    ToPredict = HeteroData()

    ToPredict["book"].node_id = torch.arange(len(result_book_features))
    ToPredict["user"].node_id = torch.arange(len(user_id_to_node_id) + 10)


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
    
def predict_for_user(user_id):
    # Get the user node ID for the specific user.
    # Ensure the user_id exists in user_id_to_node_id or handle it accordingly.
    user_node_id = user_id_to_node_id.get(user_id, len(user_id_to_node_id))

    # Create edges between the user and all books for prediction.
    all_book_node_ids = torch.arange(len(result_book_features), dtype=torch.long).to(device)
    user_node_ids = torch.full((len(all_book_node_ids),), user_node_id, dtype=torch.long).to(device)
    
    # Create edge index for the specific user interacting with every book.
    edge_index_user = torch.stack([user_node_ids, all_book_node_ids], dim=0)
   

    # Update the graph data with these edges (for prediction purposes only).
    newdata = HeteroData()
    newdata["book"].node_id = torch.arange(len(result_book_features))
    newdata["user"].node_id = torch.arange(len(user_id_to_node_id) + 10)
    newdata["book"].x = tensor(result_book_features.values, dtype=torch.float).to(device)

    newdata["user", "review", "book"].edge_index = edge_index_user

    print(newdata.validate())
    print(newdata["user", "review", "book"].edge_index.shape)
    # Ensure data is on the correct device.
    
    newdata = T.ToUndirected()(newdata)
    assert newdata["user"].num_nodes == len(user_id_to_node_id) + 10
    assert newdata["book"].num_features == 23
    assert newdata["user"].num_features == 0
    newdata = newdata.to(device)
    
    # Make predictions using the trained model.
    gatModel_3.eval()  # Set the model to evaluation mode.
    with torch.no_grad():
        # Perform a forward pass with the model.
        print("ovde?")

        pred = gatModel_3(newdata, edge_index_user)
        print("ovde?")
        pred = torch.sigmoid(pred)  # Convert logits to probabilities.
    
    print("ovde?")

    # `pred` now contains the probabilities of interaction for the user with each book.
    # Convert predictions to a list with book indices and their respective probabilities.
    predictions = list(zip(all_book_node_ids.cpu().numpy(), pred.cpu().numpy()))

    # Sort predictions by probability in descending order for top recommendations.
    predictions.sort(key=lambda x: x[1], reverse=True)

    # Print or return the top recommendations.
    print(f"Top recommendations for user {user_id}:")
    for book_id, probability in predictions[:100]:
        print(f"Book ID: {book_id}, Probability: {probability:.4f}")

    return predictions