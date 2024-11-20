class Book:
    def __init__(self, row):
        self.id = int(row["index_column"])
        self.title = str(row["Title"])
        self.author = str(row["authors_x"])
        self.description = str(row["description_x"])
        self.publisher = str(row["publisher_x"])
        self.category = str(row["categories_x"])
        self.publishedDate = str(row["publishedDate_x"])
        self.infoLink = str(row["infoLink"])
        self.countReviews = int(row["#reviews"])
        self.rating = float(row["Average Score"])
        self.image = str(row["image_x"])

    # Овој метод го користиме за да го претвориме објектот во речник за мала картичка 
    def to_dict_small(self):
        return {
            "id": self.id,
            "title": self.title,
            "author": self.author,
            "category": self.category,
            "countReviews": self.countReviews,
            "rating": self.rating,
            "image": self.image,
        }

    # Овој метод го користиме за да го претвориме објектот во речник за голема картичка (страна со повеќе детали)
    def to_dict_large(self):
        return {
            "id": self.id,
            "title": self.title,
            "author": self.author,
            "description": self.description,
            "publisher": self.publisher,
            "category": self.category,
            "publishedDate": self.publishedDate,
            "infoLink": self.infoLink,
            "countReviews": self.countReviews,
            "rating": self.rating,
            "image": self.image,
        }

    def __repr__(self):
        return f"{self.title} by {self.author} with rating {self.rating}"
