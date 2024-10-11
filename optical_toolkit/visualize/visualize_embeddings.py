import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler


def visualize_embeddings(X):
    """
    Given a numpy array of images X, flatten the images
    and plot a 2D embedding using t-SNE.

    Parameters:
    X (numpy array): Array of images of shape (num_images, height, width, channels)
                     or (num_images, height, width) for grayscale images.

    Returns:
    None
    """

    # Get number of images and flatten each image
    num_images = X.shape[0]
    image_size = np.prod(
        X.shape[1:]
    )  # height * width * channels (or height * width for grayscale)
    flat_images = X.reshape(num_images, image_size)

    # Apply t-SNE to reduce dimensionality to 2D
    tsne = TSNE(n_components=2, random_state=42)
    embedding = tsne.fit_transform(flat_images)
    embedding = MinMaxScaler().fit_transform(embedding)

    # Plot the 2D embedding
    plt.figure(figsize=(8, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], s=5, c="black")
    plt.title("2D Embedding of Images using t-SNE")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()
    plt.savefig("examples/tsne_embedding.png", dpi=300)

    return embedding
