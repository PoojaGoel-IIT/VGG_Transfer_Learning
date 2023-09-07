# organize dataset into a useful structure
from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random
# create directories
dataset_home = 'dataset_dogs_vs_cats/'
subdirs = ['train/', 'test/']
for subdir in subdirs:
 # create label subdirectories
 labeldirs = ['panda/', 'parrot/']
 for labldir in labeldirs:
     newdir = dataset_home + subdir + labldir
     makedirs(newdir, exist_ok=True)
# seed random number generator
seed(1)
# define ratio of pictures to use for validation
val_ratio = 0.2
# copy training dataset images into subdirectories
src_directory = 'train/'
for file in listdir(src_directory):
 src = src_directory + '/' + file
 dst_dir = 'train/'
 if random() < val_ratio:
     dst_dir = 'test/'
 if file.startswith('panda'):
     dst = dataset_home + dst_dir + 'panda/'  + file
     copyfile(src, dst)
 elif file.startswith('parrot'):
     dst = dataset_home + dst_dir + 'parrot/'  + file
 copyfile(src, dst)