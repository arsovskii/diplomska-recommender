import requests
from requests.adapters import HTTPAdapter, Retry
from constants import ACCESS_TOKEN
import json
from bs4 import BeautifulSoup
import pandas as pd

s = requests.Session()

retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])

s.mount("https://", HTTPAdapter(max_retries=retries))

df = pd.read_csv("../../data/newBooks2.csv")


def get_wikipedia_page(title: str):
    language_code = "en"

    number_of_results = 1
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "User-Agent": "book scraper (arsovskigames@gmail.com))",
    }

    base_url = "https://api.wikimedia.org/core/v1/wikipedia"
    endpoint = f"/en/search/title"

    # endpoint_page = f"/en/page/hobbit"
    url = base_url + endpoint
    parameters = {
        "language": "en",
        "q": {title},
        "limit": number_of_results,
    }
    response = s.get(url, headers=headers, params=parameters)
    response_json = response.json()
    # print(json.dumps(response_json, indent=4))
    if len(response_json["pages"]) == 0:
        return None
    return response_json["pages"][0]["key"]


def get_wikipedia_data(i: int, title: str):
    language_code = "en"

    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "User-Agent": "book scraper (arsovskigames@gmail.com))",
    }

    base_url = "https://api.wikimedia.org/core/v1/wikipedia"
    endpoint = f"/en/page/{title}/html"

    # endpoint_page = f"/en/page/hobbit"
    url = base_url + endpoint

    response = s.get(url, headers=headers)

    soup = BeautifulSoup(response.text, "html.parser")

    genres = []
    for th in soup.findAll("th", {"class": "infobox-label"}):

        if th.text == "Genre":
            soup.find
            for genre in th.find_next_sibling("td").findAll("a"):
                genres.append(genre.text)
    if len(genres) == 0:
        return
    df.at[i, "genres_wiki"] = genres
    # print(genres)


# wiki_wiki = wikipediaapi.Wikipedia('book scraper (arsovskigames@gmail.com)', 'en')

# page_py = wiki_wiki.page('Python_(programming_language)')
size = len(df)

# for i, row in df.iterrows():
#     print(f"{i}/{size}")
    
#     title = get_wikipedia_page(row["Title"])
#     if title is not None:
#         get_wikipedia_data(i, title)
