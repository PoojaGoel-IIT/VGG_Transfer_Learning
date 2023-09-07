from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
def plot_image_with_label(image,label):
    new_image=[]
    for i in range(len(image)):
        img = plt.imread(image[i])
        plt.imshow(img)
        plt.title(label[i])
        
        
        # Create the directory if it doesn't exist
        if not os.path.exists("labeled_images"):
            os.makedirs("labeled_images")
# Generate the new filename by combining the label and the file extension
        image_path=image[i]
        
        
         
        new_filename = f"{label[i]}{os.path.splitext(image_path)[1]}"
        print(new_filename)
# Construct the path to the new file
        new_file_path = os.path.join("labeled_images", new_filename)
        print(new_file_path)
# Save the image with the label as the filename
        plt.savefig(new_file_path)

# Print a message indicating that the file has been saved
        
plot_image_with_label(["sample_image.jpg"],["parrot"])