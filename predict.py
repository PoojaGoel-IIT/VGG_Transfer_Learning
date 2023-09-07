from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorboard.plugins import projector
import tensorflow as tf
import numpy as np
import os
import sys
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import io
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import time
import numpy as np

physical_devices=tf.config.list_physical_devices("GPU")
# load and prepare the image
def load_image(filename):
    # load the image
    img = load_img(filename, target_size=(224, 224))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    img = img.reshape(1,224, 224, 3)
    # center pixel data
    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]
    return img

# load an image and predict the class
def run_example():
    # load the image
    img = load_image('sample_image.jpg')
    # load model
    model = load_model('final_model.h5')
    # predict the class
    result = model.predict(img)
    print(result[0])
    
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
        
        
         
        new_filename = f"{label[i]+str(i)}{os.path.splitext(image_path)[1]}"
        
# Construct the path to the new file
        new_file_path = os.path.join("labeled_images", new_filename)
        new_image.append(new_file_path)
        
# Save the image with the label as the filename
        plt.savefig(new_file_path)
    return new_image
    
  

        
    

# entry point, run the example


# Load the model
model = tf.keras.models.load_model('final_model.h5')

# Load the test set
test_dir = 'Test'
test_files = os.listdir(test_dir)

# Create a summary writer for TensorBoard

writer = tf.summary.create_file_writer('logs/predicted')
class_names=['panda','parrot']
# Loop over each test image and make a prediction
image=[]
label=[]
file=[]
for filename in test_files:
    # Load the image
    filename='Test/'+filename
    img = load_image(filename)
     
    file.append(filename)
    # Make the prediction
    
    prediction = model.predict(img)
   
    prediction_class=np.where(prediction > 0.5, 1, 0).flatten()
    prediction_class_final=prediction_class.item()
    label.append(class_names[prediction_class_final])
   
    img=img.flatten().reshape(224,224,3)
    image.append(img)
   
new_image=plot_image_with_label(file,label)
 

datagen = ImageDataGenerator(rescale=1.0/255.0)
img_list = datagen.flow_from_directory('labeled_images/',class_mode='binary',batch_size=46,
		target_size=(200, 200))
batch_image,batch_label = img_list.next()
    
with writer.as_default():
        #for i in range(len(label)):
        tf.summary.image("image",batch_image,max_outputs=46,step=0)
    

    
    
fig, axs = plt.subplots(5, 5, figsize=(10, 10))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i, ax in enumerate(axs.flat):
    ax.imshow(image[i])
    ax.axis('off')
    ax.set_title("Predicted: {}".format(label[i]), fontsize=8)
plt.show() 
    # Write the prediction to TensorBoard
#with writer.as_default():
        #for i in range(len(label)):
            #tf.summary.image("image",data= new_image_path_list,max_outputs=len(label),step=0)
        # tf.summary.image("image",image,max_outputs=len(label),label=step=0) 
        #tf.summary.text("text",label,step=0)   
    
    
   
    
   
        
        
        
