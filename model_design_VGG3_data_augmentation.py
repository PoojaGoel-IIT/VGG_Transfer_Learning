# baseline model with data augmentation for the dogs vs cats dataset
import sys
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import time 
import tensorflow as tf
# define cnn model
def define_model():
 model = Sequential()
 model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
 model.add(MaxPooling2D((2, 2)))
 model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
 model.add(MaxPooling2D((2, 2)))
 model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
 model.add(MaxPooling2D((2, 2)))
 model.add(Flatten())
 model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
 model.add(Dense(1, activation='sigmoid'))
 # compile model
 opt = SGD(lr=0.001, momentum=0.9)
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
 
# run the test harness for evaluating a model
def run_test_harness():
 # define model
 model = define_model()
 # create data generators
 train_datagen = ImageDataGenerator(rescale=1.0/255.0,
 width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
 test_datagen = ImageDataGenerator(rescale=1.0/255.0)
 # prepare iterators
 train_it = train_datagen.flow_from_directory('dataset_panda_vs_parrot/train/',
 class_mode='binary', batch_size=64, target_size=(200, 200))
 test_it = test_datagen.flow_from_directory('dataset_panda_vs_parrot/test/',
 class_mode='binary', batch_size=64, target_size=(200, 200))
 # fit model
 start_time=time.time()
 history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
 validation_data=test_it, validation_steps=len(test_it), epochs=50, verbose=0)
 end_time=time.time()
 train_writer=tf.summary.create_file_writer("logs/train_VGG3_dataaugmentation/")
 test_writer=tf.summary.create_file_writer("logs/test_VGG3_dataaugmentation/")
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
 
 # evaluate model
 _, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
 print("Training time: {:.2f} seconds".format(end_time-start_time))
 print("Training Loss: {:.4f}".format(np.mean(history.history['loss'])))
 print("Training accuracy: {:.4f}".format(np.mean(history.history['accuracy']*100)))
 print("Testing accuracy:{:.3f}" .format (acc * 100.0))
 print(model.summary())
 # learning curves
 summarize_diagnostics(history)
 
# entry point, run the test harness
run_test_harness()