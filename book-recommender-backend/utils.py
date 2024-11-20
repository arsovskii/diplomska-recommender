import re
import pandas as pd
from models import Book
from config import BOOKS_CSV_PATH
from gnn.gnn import train_with_user_ratings


def append_fife(url):
    """ Додаваме параметар fife=w800 на крајот на URL-то ако не постои, за да добиеме слика во поголема резолуција"""
    if not url:
        return "/invalid.jpg"  # Доколку не постои слика
    if "&fife=w800" not in url:
        return f"{url}&fife=w800"
    return url


def clean_value(value):
    """ Функција за чистење на вредностите од празни множества и низи """
    if pd.isna(value) or value == "[]":
        return ""

    return re.sub(r"[\[\]']", "", value)


def clean_category(row):
    """ Функција за чистење на категориите на книгите """
    cleaned_category = clean_value(row["categories_x"])
    cleaned_genres_wiki = clean_value(row["genres_wiki"])

    return cleaned_category if cleaned_category else cleaned_genres_wiki


all_books = pd.read_csv(BOOKS_CSV_PATH)
all_books.rename(columns={"index": "index_column"}, inplace=True)
# all_books = all_books[all_books["image_x"].notna()]

all_books = all_books.fillna("")
all_books["book_node_id"] = all_books["book_node_id"].astype(int)
all_books["categories_x"] = all_books.apply(clean_category, axis=1)
# менување на линкот од .nl во .com
all_books["infoLink"] = all_books["infoLink"].apply(
    lambda x: x.replace(".nl/", ".com/")
)

most_rated = (
    all_books[all_books["image_x"] != ""]
    .sort_values(by="#reviews", ascending=False)
    .head(100)
)
all_books["image_x"] = all_books["image_x"].apply(append_fife)



def get_random_top_books(number: int):
    """ Враќа случајно избрани книги од најпопуларните """
    sampled = most_rated.sample(number)

    books = [Book(row).to_dict_small() for _, row in sampled.iterrows()]

    return books


def get_book(book_id: int):
    """ Враќа книга според book_id """
    book_id = int(book_id)

    book = all_books[all_books["index_column"] == book_id].iloc[0]

    return Book(book).to_dict_large()


def search_book_by_title(title: str):
    """ Пребарува книги според наслов """
    books = (
        all_books[all_books["Title"].str.contains(title, case=False)]
        .sort_values(by="#reviews", ascending=False)
        .head(10)
    )

    return [Book(row).to_dict_small() for _, row in books.iterrows()]


def get_nodeid_from_bookid(book_id: int):
    """ Претвора book_id во node_id за во графот """
    book_id = int(book_id)

    book = all_books[all_books["index_column"] == book_id].iloc[0]

    return book["book_node_id"]


def get_bookid_from_nodeid(node_id: int):
    """ Претвора node_id во book_id за во базата на книги """
    node_id = int(node_id)

    book = all_books[all_books["book_node_id"] == node_id].iloc[0]

    return book["index_column"]


def get_books_from_nodeids(node_ids):
    """ Враќа книги од листа на node_id """
    books = []

    for dict_row in node_ids:

        node_id = int(list(dict_row.keys())[0])

        prediction = dict_row[node_id]

        book = all_books[all_books["book_node_id"] == int(node_id)].iloc[0]
        dict_book = Book(book).to_dict_small()

        dict_book["prediction"] = prediction

        books.append(dict_book)
    return books


def retrieve_predicted_recommendations(ratings):
    """ Враќа препораки за книги врз основа на рејтинзите на корисникот """
    keys = ratings.keys()
    to_send = []
    for key in keys:
        if ratings[key] >= 3:
            to_send.append(get_nodeid_from_bookid(key))

    predictions, diverse_predictions = train_with_user_ratings(to_send)

    books = get_books_from_nodeids(diverse_predictions)
    sorted_books = sorted(books, key=lambda x: x["prediction"], reverse=True)
    return sorted_books
