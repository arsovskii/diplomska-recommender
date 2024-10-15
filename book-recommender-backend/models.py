class Book:
    def __init__(self, localId, title, author, category, countReviews, rating, image):
        self.id = localId
        self.title = title
        self.author = author
        self.category = category
        self.countReviews = countReviews
        self.rating = rating
        self.image = image

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "author": self.author,
            "category": self.category,
            "countReviews": self.countReviews,
            "rating": self.rating,
            "image": self.image,
        }

    def __repr__(self):
        return f"{self.title} by {self.author} with rating {self.rating}"
