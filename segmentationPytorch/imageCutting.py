import os
import re
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
import cv2
import numpy as np
from PIL import Image

# Function to create grayscale mosaic

newSize = [1200,1200]

# Load the image
# image_path = 'C:/Users/Matus/Desktop/FEI/dataset_OIA_DDR/DDR-dataset/lesion_segmentation/test/image/007-3454-200.jpg'  # Replace with your image path
# image_path = 'C:/Users/Matus/Desktop/FEI/dataset_OIA_DDR/DDR-dataset/lesion_segmentation/test/image/007-3489-200.jpg'
# image_path = 'C:/Users/Matus/Desktop/FEI/dataset_OIA_DDR/DDR-dataset/lesion_segmentation/test/image/007-3494-200.jpg'
# image_path = 'C:/Users/Matus/Desktop/FEI/dataset_OIA_DDR/DDR-dataset/lesion_segmentation/test/image/007-3722-200.jpg'
# image_path = 'C:/Users/Matus/Desktop/FEI/dataset_OIA_DDR/DDR-dataset/lesion_segmentation/test/image/20170627170651362.jpg'
image_path = 'C:/Users/Matus/Desktop/FEI/dataset_OIA_DDR/DDR-dataset/lesion_segmentation/test/image/20170505161143089.jpg'

# image_mask_path = 'C:/Users/Matus/Desktop/FEI/dataset_OIA_DDR/DDR-dataset/lesion_segmentation/test/label/EX/007-3489-200.tif'
# image_mask_path = 'C:/Users/Matus/Desktop/FEI/dataset_OIA_DDR/DDR-dataset/lesion_segmentation/test/label/EX/007-3494-200.tif'
# image_mask_path = 'C:/Users/Matus/Desktop/FEI/dataset_OIA_DDR/DDR-dataset/lesion_segmentation/test/label/EX/007-3722-200.tif'
image_mask_path = 'C:/Users/Matus/Desktop/FEI/dataset_OIA_DDR/DDR-dataset/lesion_segmentation/test/label/SE/20170627170651362.tif'

image = cv2.imread(image_path)
image_mask = cv2.imread(image_mask_path)


def center_crop(image, size):
    target_width, target_height = size
    width, height = image.size

    # Determine the shortest side for cropping to 1:1 aspect ratio
    crop_size = min(width, height)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size

    # Crop the image to 1:1 aspect ratio
    cropped_image = image.crop((left, top, right, bottom))

    # Resize the cropped image to the desired resolution
    resized_image = cropped_image.resize((target_width, target_height))

    return resized_image




blue_channel = image[:, :, 0]
green_channel = image[:, :, 1]
red_channel = image[:, :, 2]

gray_red = Image.fromarray(red_channel).convert("L")
gray_green = Image.fromarray(green_channel).convert("L")
gray_blue = Image.fromarray(blue_channel).convert("L")

gray_red = Image.fromarray(red_channel).convert("L")
gray_green = Image.fromarray(green_channel).convert("L")
gray_blue = Image.fromarray(blue_channel).convert("L")

# Convert original image to PIL format
original_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


original_pil = center_crop(original_pil, newSize)

original_np = np.array(original_pil)

# Convert RGB to BGR (OpenCV uses BGR format)
original_np = cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR)



def draw_moving_rectangles(image, rect_size=120, rect_thickness=1):
    # Get image dimensions
    height, width, _ = image.shape

    # Step size for moving rectangles (half of rectangle size)
    step_size = rect_size // 2

    # Colors for alternating rectangles
    colors = [(0, 255, 0),  # Green in BGR
              (255, 0, 0)]  # Red in BGR

    # Loop through the image to draw rectangles
    for y in range(0, height, step_size):
        for x in range(0, width, step_size):
            # Define the top-left and bottom-right corners of the rectangle
            top_left = (x, y)
            bottom_right = (x + rect_size, y + rect_size)

            # Determine the color: alternate based on the sum of x and y indices
            color_index = ((x // step_size) + (y // step_size)) % 2
            rect_color = colors[color_index]

            # Make sure the rectangle doesn't go out of the image boundary
            if bottom_right[0] <= width and bottom_right[1] <= height:
                cv2.rectangle(image, top_left, bottom_right, rect_color, rect_thickness)

    # Show the resulting image
    plt.axis('off')
    plt.imshow(image)
    plt.savefig('test.png')
    plt.show()

rct_size =  int(newSize[1]/6)

def draw_colored_rectangles(image, color, rect_size=rct_size, rect_thickness=3):
    """Draws rectangles of a single color (red or blue) on the image."""
    height, width, _ = image.shape
    step_size = rect_size // 2

    # Create a copy of the original image
    image_copy = image.copy()

    for y in range(0, height, step_size):
        for x in range(0, width, step_size):
            if ((x // step_size) + (y // step_size)) % 2 == (0 if color == 'red' else 1):
                top_left = (x, y)
                bottom_right = (x + rect_size, y + rect_size)
                if bottom_right[0] <= width and bottom_right[1] <= height:
                    rect_color = (0, 0, 255) if color == 'red' else (255, 0, 0)  # BGR format
                    cv2.rectangle(image_copy, top_left, bottom_right, rect_color, rect_thickness)

    return image_copy


# Example usage
# draw_moving_rectangles(original_np)

# Draw red rectangles and save the image
red_image = draw_colored_rectangles(original_np, color='red')
cv2.imwrite('red_rectangles.png', red_image)

# Draw blue rectangles and save the image
blue_image = draw_colored_rectangles(original_np, color='blue')
cv2.imwrite('blue_rectangles.png', blue_image)

# Display the images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.axis('off')
plt.imshow(cv2.cvtColor(red_image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 2, 2)
plt.axis('off')
plt.imshow(cv2.cvtColor(blue_image, cv2.COLOR_BGR2RGB))

plt.savefig('test.png')
plt.show()
