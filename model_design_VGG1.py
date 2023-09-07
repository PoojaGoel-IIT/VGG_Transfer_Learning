# baseline model for the dogs vs cats dataset
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
#tf.config.experimental.set_memory_growth(physical_devices[0],True)


# define cnn modelpip
def define_model():
 model = Sequential()
 model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
 model.add(MaxPooling2D((2, 2)))
 model.add(Flatten())
 model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
 model.add(Dense(1, activation='sigmoid'))
 # compile model
 opt = SGD(learning_rate=0.001, momentum=0.9)
 model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
 return model
 
# plot diagnostic learning curves
def summarize_diagnostics(history):
 # plot loss
 pyplot.subplot(211)
 pyplot.title('Cross Entropy Loss')
 pyplot.plot(history.history['loss'], color='blue', label='train')
 pyplot.plot(history.history['val_loss'], color='orange', label='test')
 # plot accuracy
 pyplot.subplot(212)
 pyplot.title('Classification Accuracy')
 pyplot.plot(history.history['accuracy'], color='blue', label='train')
 pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
 # save plot to file
 filename = sys.argv[0].split('/')[-1]
 pyplot.savefig(filename + '_plot.png')
 pyplot.close()
def plot_image(train_it):
    # Get a batch of images from the generator
     batch_images, batch_labels = train_it.next()
     # Plot the images in the batch
     fig, axes = plt.subplots(nrows=8, ncols=8, figsize=(12, 12))
     for i, ax in enumerate(axes.flat):
        ax.imshow(batch_images[i])
        ax.set_title('Label: {}'.format(batch_labels[i]))
     plt.show()

 
 # run the test harness for evaluating a model
def run_test_harness():
    
    
	# define model
    model = define_model()
	# create data generator
    datagen = ImageDataGenerator(rescale=1.0/255.0)
	# prepare iterators
    train_it = datagen.flow_from_directory('dataset_panda_vs_parrot/train/',
		class_mode='binary', batch_size=64, target_size=(200, 200))
    plot_image(train_it)
    test_it = datagen.flow_from_directory('dataset_panda_vs_parrot/test/',
		class_mode='binary', batch_size=64, target_size=(200, 200))
    start_time=time.time()
	# fit model
    history = model.fit(train_it, steps_per_epoch=len(train_it),
		validation_data=test_it, validation_steps=len(test_it), epochs=5, verbose=0)
    end_time=time.time()
    batch_images, batch_labels = train_it.next()
    print(batch_images.shape)
    
    train_writer=tf.summary.create_file_writer("logs/train/")
    test_writer=tf.summary.create_file_writer("logs/test/")
    writer=tf.summary.create_file_writer("logs/images")
    with train_writer.as_default():
        for i,loss_value in enumerate(history.history['loss']):
            tf.summary.scalar("loss",loss_value,step=i)
        for i,acc_value in enumerate(history.history['accuracy']):  
            tf.summary.scalar("accuracy",acc_value, step=i)
    with test_writer.as_default():
        for i,loss_value in enumerate(history.history['val_loss']):
            tf.summary.scalar("loss",loss_value,step=i)
        for i,acc_value in enumerate(history.history['val_accuracy']):  
            tf.summary.scalar("accuracy",acc_value, step=i)
    with writer.as_default():
        tf.summary.image("Training_image",batch_images,max_outputs=64,step=0)
        
    
    # evaluate model
    _, acc = model.evaluate(test_it, steps=len(test_it), verbose=0)
    
    print("Training time: {:.2f} seconds".format(end_time-start_time))
    print("Training Loss: {:.4f}".format(np.mean(history.history['loss'])))
    print("Training accuracy: {:.4f}".format(np.mean(history.history['accuracy']*100)))
    print("Testing accuracy:{:.3f}" .format (acc * 100.0))
    print(model.summary())
	# learning curves
    summarize_diagnostics(history)
 
# entry point, run the test harness
run_test_harness()

#tensorboard --logdir logs