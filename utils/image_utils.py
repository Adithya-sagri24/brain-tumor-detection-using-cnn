# utils/image_utils.py

from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np

# Image size your model expects
IMAGE_SIZE = (150, 150)

def preprocess_image(img_path):
    """
    Load an image from disk, resize to IMAGE_SIZE,
    normalize pixel values to [0, 1], and add batch dimension.
    
    Args:
        img_path (str): Path to the image file.
    
    Returns:
        np.ndarray: Preprocessed image array with shape (1, 150, 150, 3)
    """
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Batch dimension
    return img_array
