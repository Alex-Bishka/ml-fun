import struct
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


def load_images(file_path: str) -> np.array:
    """
    Load MNIST images from a ubyte file,

    Returns a numpy array of images with shape (num_images, rows, cols)
    """
    with open(file_path, 'rb') as f:
        header = f.read(16)
        magic, num_images, rows, cols = struct.unpack('>IIII', header)

        if magic != 2051:
            raise ValueError(f"Invalid magic number: {magic}")

        data = np.frombuffer(f.read(), dtype=np.uint8)
        images = data.reshape(num_images, rows, cols)

    return images


def load_labels(file_path: str) -> np.array:
    with open(file_path, 'rb') as f:
        header = f.read(8)
        magic, num_labels = struct.unpack('>II', header)

        if magic != 2049:
            raise ValueError(f"Invalid magic number: {magic}")

        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return labels


def visualize_image(sample_img, sample_label, ax=None):
    """
    """
    if ax is None:
        plt.figure(figsize=(3, 3))
        plt.imshow(sample_img, cmap='gray')
        plt.title(f"Label: {sample_label}")
        plt.axis('off')
        plt.show()
    else:
        ax.imshow(sample_img, cmap='gray')
        ax.set_title(f"Label: {sample_label}")
        ax.axis("off")


def get_edges(kernel, img):
    convolved = convolve2d(img, kernel, mode='same', boundary='fill')
    convolved_abs = np.abs(convolved)

    convolved_normalized = (convolved_abs - convolved_abs.min()) / (convolved_abs.max() - convolved_abs.min()) * 255
    convolved_normalized = convolved_normalized.astype(np.uint8)

    return convolved_normalized


def generate_intermediate_edge_labels(images, kernel):
    """
    """
    intermediate_labels_list = []
    for img in images:
        edges = get_edges(kernel, img)
        intermediate_labels_list.append(edges.flatten())
    
    intermediate_labels = np.array(intermediate_labels_list)
    return intermediate_labels