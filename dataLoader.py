# A Python script to load image data and prepare it for training.
# It will read data from the train_data folder.
# The groun truth values are contained in the ground_truth.csv folder. 

from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
from joblib import delayed, Parallel
import psutil

# A class for loading data
class data_loader():

    def __init__(self, train_directory, val_directory, test_directory, data_size):
        self.train_directory = train_directory
        self.val_directory = val_directory
        self.test_directory = test_directory    

    
    def load_images(self):
        train_datagen = ImageDataGenerator()#width_shift_range=0.2,height_shift_range=0.2,zoom_range=0.2,fill_mode='nearest' )
        # Need to test with class_mode = "input"
        train_generator = train_datagen.flow_from_directory(self.train_directory, 
                                target_size=(128, 128),
                                color_mode="grayscale",
                                class_mode="categorical",
                                batch_size=128,
                                shuffle =True,
                                seed=42, classes=['centred', 'hexagonal', 'noise', 'oblique','rectangular','square'])
        
        val_datagen = ImageDataGenerator()
        val_generator = val_datagen.flow_from_directory(self.val_directory,
                                target_size=(128,128),
                                color_mode="grayscale",
                                class_mode='categorical',
                                shuffle =True,seed=42,
                                batch_size=128,classes=['centred', 'hexagonal', 'noise', 'oblique','rectangular','square'])
        
        
        test_datagen = ImageDataGenerator()
        test_generator = test_datagen.flow_from_directory(self.test_directory, 
                                target_size=(128, 128),
                                color_mode="grayscale",
                                class_mode="categorical",
                                batch_size=128,classes=['centred', 'hexagonal', 'noise', 'oblique','rectangular','square'])
        
        

    

        return train_generator, val_generator, test_generator
