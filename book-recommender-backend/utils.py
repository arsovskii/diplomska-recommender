import pandas as pd
from models import Book
from config import BOOKS_CSV_PATH


def append_fife(url):
    if "&fife=w800" not in url:
        return f"{url}&fife=w800"
    return url


all_books = pd.read_csv(BOOKS_CSV_PATH)
all_books.rename(columns={"index": "index_column"}, inplace=True)
all_books = all_books[all_books["image_x"].notna()]
all_books = all_books.fillna("")
all_books["image_x"] = all_books["image_x"].apply(append_fife)
all_books["infoLink"] = all_books["infoLink"].apply(lambda x: x.replace(".nl/", ".com/"))

most_rated = all_books.sort_values(by="#reviews", ascending=False).head(100)


def get_random_top_books(number: int):
    sampled = most_rated.sample(number)

    books = [
        Book(
            row
        ).to_dict_small()
        for _, row in sampled.iterrows()
    ]
    print(books)
    return books


def get_book(book_id: int):
    book_id = int(book_id)

    book = all_books[all_books["index_column"] == book_id].iloc[0]

    return Book(
        book
    ).to_dict_large()
