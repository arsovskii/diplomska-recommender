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

# Вчитуваме сите податоци за книгите, како и моделите кои ни се истренирани

model_state = torch.load("./gnn/graphdata/gat_model_3.pth")
review_edges = torch.load("./gnn/graphdata/review_edges.pt")
review_edges = review_edges.to(device)

user_id_to_node_id = {}

with open("./gnn/graphdata/saved_dictionary.pkl", "rb") as f:
    user_id_to_node_id = pickle.load(f)

result_book_features = pd.read_csv("./gnn/graphdata/book_features.csv")

# Го формираме графот повторно

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

# Го дефинираме истиот модел
 
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
        

        edge_feat_user = x_user[edge_label_index[0]].to(dtype=x_book.dtype)
        edge_feat_book = x_book[edge_label_index[1]]

        
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

# Иницијализираме модел и оптимизатор
gatModel_3 = GATModel_3(hidden_channels=64).to(
    device
)  
optimizer = torch.optim.Adam(
    gatModel_3.parameters(),
    lr=0.00001, # понизок learning rate
)  

# Ги вчитуваме зачуваните состојби на моделот и оптимизаторот
gatModel_3.load_state_dict(model_state["model_state_dict"])
optimizer.load_state_dict(model_state["optimizer_state_dict"])


def train_with_user_ratings(user_ratings):
    """При секоја препорака, го тренираме моделот со новите рејтинзи на корисникот, базирано врз веќе истренираниот модел"""
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

    # Креирање граф за тренирање
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

    # Batch training за помалку трошење на меморија
    data_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=[-1, -1],
        neg_sampling_ratio=4.0,  
        edge_label_index=(("user", "review", "book"), edge_index_user),
        batch_size=256,  
        shuffle=True,
    )

    pos_weight = torch.tensor([4.0]).to(device)  # Задавање на тежина на позитивната класа
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    gatModel_3.train()

    for epoch in range(5):
        total_loss = total_examples = 0
        for batch in tqdm(data_loader):
            optimizer.zero_grad()

            batch = batch.to(device)
            pred = gatModel_3(batch, batch["user", "review", "book"].edge_label_index)

            # label smoothing - за да се намали претренирањето
            ground_truth = batch[("user", "review", "book")].edge_label
            smoothing = 0.1
            ground_truth = ground_truth * (1 - smoothing) + smoothing / 2

            loss = loss_fn(pred, ground_truth)

            # L2 регуларизација
            l2_lambda = 0.01
            l2_reg = torch.tensor(0.0).to(device)
            for param in gatModel_3.parameters():
                l2_reg += torch.norm(param)
            loss += l2_lambda * l2_reg

            loss.backward()

            # Gradient clipping за постабилен процес на тренирање и побрза конвергенција
            torch.nn.utils.clip_grad_norm_(gatModel_3.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()

        print(f"Epoch: {epoch+1:03d}, Loss: {total_loss / total_examples:.4f}")

    return get_predictions(user_node_id, user_ratings)

   
def get_predictions(user_node_id: int, userratings: List[int]):
    """Функција за добивање на препораки за корисникот"""
    gatModel_3.eval()

    # Креирање на графот за предвидување
    pred_data = HeteroData()

    # Ги наоѓаме сите книги кои не се оценети од корисникот
    all_book_ids = set(range(len(result_book_features)))
    rated_books = set(userratings)
    books_to_predict = list(all_book_ids - rated_books)

    # Додавање на книгите кои не се оценети во графот
    pred_data["book"].node_id = torch.arange(len(result_book_features)).to(device)
    pred_data["user"].node_id = torch.arange(len(user_id_to_node_id) + 10).to(device)
    pred_data["book"].x = tensor(result_book_features.values, dtype=torch.float).to(
        device
    )

    # Додавање на релациите од тренинг графот
    edge_list = []
    if userratings:
        existing_user_nodes = torch.full(
            (len(userratings),), user_node_id, dtype=torch.long
        ).to(device)
        existing_book_nodes = torch.tensor(userratings, dtype=torch.long).to(device)
        existing_edges = torch.stack([existing_user_nodes, existing_book_nodes], dim=0)
        edge_list.append(existing_edges)

    # Додавање на релациите од тренинг графот
    edge_list.append(data["user", "review", "book"].edge_index.to(device))

    # Додавање на релациите од рецензиите
    pred_data["user", "review", "book"].edge_index = torch.cat(edge_list, dim=1)

   
    pred_data = T.ToUndirected()(pred_data)
    pred_data = pred_data.to(device)


    batch_size = 512
    predictions = []
    
    # Предвидување на рејтинзите за книгите
    with torch.no_grad():
        # Batch предвидувања, предвидување на рејтинзите за сите книги
        for i in range(0, len(books_to_predict), batch_size):
            batch_books = books_to_predict[i : i + batch_size]
            batch_users = torch.full(
                (len(batch_books),), user_node_id, dtype=torch.long
            ).to(device)
            batch_books_tensor = torch.tensor(batch_books, dtype=torch.long).to(device)
            batch_edge_index = torch.stack([batch_users, batch_books_tensor], dim=0)

            # Предвидување на рејтинзите
            out = gatModel_3(pred_data, batch_edge_index)

            # Пресметување на веројатностите
            temperature = 1.5
            scaled_logits = out / temperature
            probabilities = torch.sigmoid(scaled_logits)

            # Зачувување на предвидувањата за секоја книга во листа, вклучувајќи ги и карактеристиките на книгите и id на книгите
            for book_id, prob in zip(batch_books, probabilities.cpu().numpy()):
                book_features = result_book_features.iloc[book_id]
                predictions.append(
                    {
                        "book_id": book_id,
                        "probability": float(prob),
                        "features": book_features,
                    }
                )

    # Сортирање на предвидувањата според веројатноста
    predictions.sort(key=lambda x: x["probability"], reverse=True)

    # Диверзификација на препораките
    diverse_predictions = []
    feature_memory = []
    similarity_threshold = 0.8

    # Пресметување на сличноста помеѓу карактеристиките на книгите
    def compute_feature_similarity(features1, features2):
        return float(
            torch.cosine_similarity(
                torch.tensor(features1.values), torch.tensor(features2.values), dim=0
            )
        )

    for pred in predictions:
        # Прескокнување на книгите со ниска веројатност
        if pred["probability"] < 0.5 :
            continue

        # Проверка дали книгата е диверзификувана
        is_diverse = True
        for selected_features in feature_memory:
            # Пресметување на сличноста помеѓу карактеристиката на моменталната книга и претходно ставените уникатни книги
            similarity = compute_feature_similarity(pred["features"], selected_features)
            
            # Ако сличноста е поголема од прагот, значи книгата не е уникатна
            if similarity > similarity_threshold:
                is_diverse = False
                break

        if is_diverse:
            # Додавање на уникатната книга во листата
            diverse_predictions.append({pred["book_id"]: pred["probability"]})
            feature_memory.append(pred["features"])

            # Прекинување на додавањето на книги ако се добиени 10 уникатни книги
            if len(diverse_predictions) >= 10:
                break

    # Додавање на книги со висока веројатност ако нема доволно уникатни книги
    if len(diverse_predictions) < 10:
        for pred in predictions:
            if len(diverse_predictions) >= 10:
                break

            if pred["probability"] > 0.95:
                continue

            if pred["book_id"] not in [list(x.keys())[0] for x in diverse_predictions]:
                diverse_predictions.append({pred["book_id"]: pred["probability"]})

    return predictions, diverse_predictions
