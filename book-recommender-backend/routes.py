from flask import Flask, request, Blueprint, jsonify
import ast
import pandas as pd

from utils import (
    get_book,
    get_random_top_books,
    retrieve_predicted_recommendations,
    search_book_by_title,
)

api_routes = Blueprint("api", __name__)

num_recommendations = 10    

@api_routes.route("/api/recommend", methods=["POST"])
def recommend_books():
    # Враќаме 10 најпопуларни книги 
    
    
    rand_books = []
    try:
        content_type = request.headers.get("Content-Type")

        if content_type == "application/json":
            
            
            json = request.json
            genres = json.get("genres")

            if genres:
                # Пример на json што се праќа: {"genres": ["Fantasy", "Adventure"]}
                genres_list = ast.literal_eval(genres)
                rand_books = get_random_top_books(num_recommendations, genres_list)
            else:
                rand_books = get_random_top_books(num_recommendations)

          

            return jsonify({"books": rand_books})
        else:
            raise ValueError("Content-Type must be application/json")

    except Exception:
       
        return jsonify({"books": rand_books})

    
    # return jsonify({"books": rand_books})


@api_routes.route("/api/book", methods=["POST"])
def book_by_id():
    # Враќа книга според book_id

    try:
        content_type = request.headers.get("Content-Type")

        if content_type == "application/json":
            # Пример на json што се праќа: {"book_id": 1}
            
            json = request.json
            book_id = json.get("book_id")
            book = get_book(book_id)

            return jsonify({"book": book})
        else:
            raise ValueError("Content-Type must be application/json")

    except Exception as e:
        return jsonify({"error": e.args[0]})


@api_routes.route("/api/search", methods=["POST"])
def search_books():

    try:
        content_type = request.headers.get("Content-Type")

        if content_type == "application/json":
            # Пример на json што се праќа: {"title": "Harry Potter"}

            json = request.json
            title = json.get("title")
            books = search_book_by_title(title)

            return jsonify({"books": books})
        else:
            raise ValueError("Content-Type must be application/json")

    except Exception as e:
        return jsonify({"error": e.args[0]})


@api_routes.route("/api/makeRecommendation", methods=["POST"])
def make_recommendation():
   
    try:
        content_type = request.headers.get("Content-Type")

        if content_type == "application/json":
            
            # Пример на json што се праќа: {"ratings": {"book_id": 1, "rating": 5,"book_id": 2, "rating": 3}}
            json = request.json
            predictions = retrieve_predicted_recommendations(json["ratings"])

            return jsonify({"preds": predictions})
        else:
            raise ValueError("Content-Type must be application/json")

    except Exception as e:
        return jsonify({"error": e.args[0]})
