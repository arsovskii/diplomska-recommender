from flask import Flask, Blueprint, jsonify
from utils import get_random_top_books

api_routes = Blueprint("api", __name__)


@api_routes.route("/api/recommend", methods=["POST"])
def recommend_books():
    # Here you can add more logic like reading user_id or preferences from request
    rand_books = get_random_top_books(10)
    return jsonify({"books": rand_books})
