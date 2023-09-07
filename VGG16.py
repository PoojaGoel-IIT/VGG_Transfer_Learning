import sys
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import time
import numpy as np
# define cnn model
def define_model():
 # load model
 model = VGG16(include_top=False, input_shape=(224, 224, 3))
 # mark loaded layers as not trainable
 for layer in model.layers:
     layer.trainable = False
 # add new classifier layers
 flat1 = Flatten()(model.layers[-1].output)
 class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
 output = Dense(1, activation='sigmoid')(class1)
 # define new model
 model = Model(inputs=model.inputs, outputs=output)
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
 # create data generator
 datagen = ImageDataGenerator(featurewise_center=True)
 # specify imagenet mean values for centering
 datagen.mean = [123.68, 116.779, 103.939]
 # prepare iterator
 train_it = datagen.flow_from_directory('dataset_panda_vs_parrot/train/',
 class_mode='binary', batch_size=64, target_size=(224, 224))
 test_it = datagen.flow_from_directory('dataset_panda_vs_parrot/test/',
 class_mode='binary', batch_size=64, target_size=(224, 224))
 # fit model
 start_time=time.time()
 history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
 validation_data=test_it, validation_steps=len(test_it), epochs=10, verbose=1)
 end_time=time.time()
 import tensorflow as tf
 
 train_writer=tf.summary.create_file_writer("logs/train_VGG16/")
 test_writer=tf.summary.create_file_writer("logs/test_VGG16/")
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