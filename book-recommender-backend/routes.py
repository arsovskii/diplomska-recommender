from flask import Flask, request, Blueprint, jsonify
from utils import get_book, get_random_top_books, search_book_by_title

api_routes = Blueprint("api", __name__)


@api_routes.route("/api/recommend", methods=["POST"])
def recommend_books():
    # Here you can add more logic like reading user_id or preferences from request
    rand_books = get_random_top_books(10)
    return jsonify({"books": rand_books})


@api_routes.route("/api/book", methods=["POST"])
def book_by_id():
    # Here you can add more logic like reading user_id or preferences from request
    try:
        content_type = request.headers.get("Content-Type")

        if content_type == "application/json":
            json = request.json
            book_id = json.get("book_id")
            book = get_book(book_id)
            print(book)
            return jsonify({"book":book})
        else:
            raise ValueError("Content-Type must be application/json")

    except Exception as e:
        return jsonify({"error": e.args[0]})

@api_routes.route("/api/search", methods=["POST"])
def search_books():
   
   
    try:
        content_type = request.headers.get("Content-Type")

        if content_type == "application/json":
            json = request.json
            title = json.get("title")
            books = search_book_by_title(title)
            

            return jsonify({"books": books})
        else:
            raise ValueError("Content-Type must be application/json")

    except Exception as e:
        return jsonify({"error": e.args[0]})