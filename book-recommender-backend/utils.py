import pandas as pd
from models import Book
from config import BOOKS_CSV_PATH


all_books = pd.read_csv("./data/books.csv")
most_rated = all_books.sort_values(by="#reviews", ascending=False).head(100)
most_rated = most_rated[most_rated["image_x"].notna()]
most_rated = most_rated.fillna("")


def append_fife(url):
    if '&fife=w800' not in url:
        return f"{url}&fife=w800"
    return url

most_rated["image_x"] = most_rated["image_x"].apply(append_fife)


def get_random_top_books(number: int):
    sampled = most_rated.sample(number)

    books = [
        Book(
            row["index"],
            row["Title"],
            row["authors_x"],
            row["categories_x"],
            row["#reviews"],
            row["Average Score"],
            row["image_x"],
        ).to_dict()
        for _, row in sampled.iterrows()
    ]
    
    return books
