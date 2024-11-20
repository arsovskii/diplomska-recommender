import requests
from requests.adapters import HTTPAdapter, Retry

import pandas as pd

import json

url = "https://www.googleapis.com/books/v1/volumes"

s = requests.Session()

retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])

s.mount("https://", HTTPAdapter(max_retries=retries))

df = pd.read_csv("../data/books.csv")


def search_book(query, i):
    title = query["Title"]
    params = {
        "q": title,
        "maxResults": 1,
        "orderBy": "relevance",
    }

    try:
        response = s.get(url, params=params)
        response.raise_for_status()  
        books = response.json().get("items", [])
        pretty = json.dumps(books, indent=4)
     

        volumeInfo = books[0]["volumeInfo"]

        authors = ""
        pageCount = ""
        description = ""
        categories = []

        if "authors" in volumeInfo:
            authors = volumeInfo["authors"][0]
        if "pageCount" in volumeInfo:
            pageCount = volumeInfo["pageCount"]
        if "description" in volumeInfo:
            description = volumeInfo["description"]
        if "categories" in volumeInfo:
            categories = volumeInfo["categories"]

        
        df.at[i, "authors_x"] = authors
        df.at[i, "description_x"] = description
        df.at[i, "categories_x"] = categories
        df.at[i, "pageCount"] = pageCount
        

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")

size = len(df)

for i, row in df.iterrows():
    search_book(row, i)
    print(f"{i}/{size}")
    
    

df.to_csv("newBooks.csv")
