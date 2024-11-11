import pandas as pd
import numpy as np

# Load the datasets
books = pd.read_csv("../data/newBooks6.csv")
features_description = pd.read_csv("G:\Fakultet\diplomska\description_embeddings.csv")
features_title = pd.read_csv("G:\Fakultet\diplomska\\title_embeddings.csv")

print("Number of books:", len(books))
print("Number of description features:", len(features_description))
print("Number of title features:", len(features_title))
print(features_description.head())

# Concatenate the description and title features
features =  features_title

# Filter titles containing 'Count of '
filtered_titles = books[books['Title'].str.contains('Count of ', case=False, na=False)]
print("Filtered Titles:")
print(filtered_titles)

# Get indices of filtered titles
filtered_indices = filtered_titles.index.tolist()
print("Filtered Indices:", filtered_indices)

# Filter features based on indices
filtered_features = features[features.index.isin(filtered_indices)]
print("Filtered Features:")
print(filtered_features)

# Calculate the cosine similarity matrix
similarity_matrix = np.dot(filtered_features, filtered_features.T)  # Matrix multiplication
norms = np.linalg.norm(filtered_features, axis=1)  # L2 norms for each vector
similarity_matrix /= norms[:, np.newaxis]  # Normalize by row
similarity_matrix /= norms[np.newaxis, :]  # Normalize by column

print("Cosine Similarity Matrix:")
print(similarity_matrix)

# Function to find similar titles based on a similarity threshold
def find_similar_titles(similarity_matrix, titles, threshold=0.8):
    similar_pairs = []
    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):  # Avoid duplicate pairs
            if similarity_matrix[i, j] > threshold:
                similar_pairs.append((titles.iloc[i]['Title'], titles.iloc[j]['Title'], similarity_matrix[i, j]))
    return similar_pairs

# Get similar titles
similar_titles = find_similar_titles(similarity_matrix, filtered_titles)

# Print similar titles
print("Similar Titles (above threshold):")
for title1, title2, sim in similar_titles:
    print(f"'{title1}' and '{title2}' are similar with a cosine similarity of {sim:.2f}")