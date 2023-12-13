
from PIL import Image
import numpy as np
from skimage.filters import gaussian

image_path = '/Users/xuguangyi/Desktop/eecs442/final_project/image-stitching/test1.png'
image = Image.open(image_path)

# Convert the image to RGB if it's in a different mode (like RGBA or P)
if image.mode != 'RGB':
    image = image.convert('RGB')

image_data = np.array(image)


threshold = 5
non_black_mask = np.any(image_data > threshold, axis=-1)


average_color = image_data[non_black_mask].mean(axis=0)
image_data[~non_black_mask] = average_color

# Function to smooth the boundaries with a wider range
def smooth_image_boundaries(image_data, mask):
    inverse_mask = ~mask

    alpha = gaussian(inverse_mask.astype(float), sigma=15)  # Increased sigma for a wider blur
    alpha = np.stack((alpha,)*3, axis=-1)
    smoothed_data = (alpha * average_color + (1 - alpha) * image_data).astype(np.uint8)

    return smoothed_data


smoothed_filled_data = smooth_image_boundaries(image_data, non_black_mask)


smoothed_filled_image = Image.fromarray(smoothed_filled_data)


smoothed_filled_image.save('smoothed_filled_image.jpg')
