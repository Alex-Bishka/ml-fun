import pickle
import struct
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.ndimage import rotate

vertical_kernel = np.array([
    [-1,  0,  1],
    [-1,  0,  1],
    [-1,  0,  1],
])

horizontal_kernel = np.array([
    [-1, -1, -1],
    [ 0,  0 , 0],
    [ 1,  1,  1],
])

curve_kernel = np.array([
    [  0,  0,-.5, -1,  0],
    [  0,-.5, -1,  0,  1],
    [-.5, -1,  0,  1, .5],
    [ -1,  0,  1, .5,  0],
    [  0,  1, .5,  0,  0]
])

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


def save_intermediate_labels(file_name, intermediate_label):
    file_path = f"./intermediate-labels/{file_name}"
    with open(file_path, "wb") as f:
        pickle.dump(intermediate_label, f)

    print("Intermediate label has been saved!")


def load_intermediate_labels(file_name):
    file_path = f"./intermediate-labels/{file_name}"
    with open(file_path, "rb") as f:
        labels = pickle.load(f)

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


def generate_intermediate_curve_labels(images, threshold = 80):
    """
    """
    thetas = list(range(0, 180 + 1, 10))
    curve_labels_list = []
    for img in images:
        curves = np.zeros((28, 28))
        
        for theta in thetas:
            rotated_kernel = rotate(curve_kernel, theta)
            curves += get_edges(rotated_kernel, img)

        avg_curves = curves / len(thetas)
        thresholded_avg_curves = np.where(avg_curves < threshold, 0, avg_curves)
        curve_labels_list.append(thresholded_avg_curves.flatten())

    curve_labels = np.array(curve_labels_list)
    return curve_labels
        