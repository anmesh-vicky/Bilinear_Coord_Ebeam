# A Python script to construct the model and train it
# Code for argument passing has to be written. 
# Hopefully this works out well enough that I dont have to spend too much time trying to fix issues. 

import argparse
from keras.utils import plot_model 
from Cordconv import CordConv as base_model
from dataLoader import data_loader
import time
from tqdm import tqdm
from keras.callbacks import ModelCheckpoint,TensorBoard

parser = argparse.ArgumentParser()

# Hyperparameters
parser.add_argument('--e', help="The number of epochs", type = int, default = 10)
parser.add_argument('--b', help="Batch size", type = int, default = 128)
parser.add_argument('--train', help = "The portion of data for training", type= float, default = 0.6)
parser.add_argument('--val', help = "The portion of data for validation", type = float, default = 0.2)
parser.add_argument('--test', help = "The portion of data for testing", type = float, default = 0.2)
parser.add_argument('--dropout', help = "Whether to include dropout", type = float, default = .2)
parser.add_argument('--kernel', help = "Kernel size", type = int, default = 5)

# Optimizer set up
parser.add_argument('--optimizer', help="The optimizer to use", type = str, default = 'adam')
parser.add_argument('--learning_rate', help="the learning rate", type = float, default = '0.0001')
parser.add_argument('--momentum', help = "The momentum", type = float, default = 0.9)
parser.add_argument('--weight_decay', help = "Weight decay factor", type = float, default = 0)
parser.add_argument('--seed', help = "Seed for random initializations", type = float, default = 43)

# Data flow
parser.add_argument('--train_data_dir', help = "Directory containing training data", type = str, default = 'data/train')
parser.add_argument('--val_data_dir', help = "Directory for validation data", type =str, default = "data/val")
parser.add_argument('--test_data_dir', help = "Directory for test data", type = str, default = "data/test")
args = parser.parse_args()

# Build model
model = base_model(learning_rate = args.learning_rate,dropout = args.dropout, kernel_size = args.kernel)
CNN = model.build_model()
#filepath = "weights-imp-{epoch:02d}-{val_acc:.2f}.hdf5"
filepath="weights_ve_res-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
# Load the data
# CNN.load_weights("weights_ve_res-improvement-32-0.83.hdf5")

dataLoader = data_loader(args.train_data_dir, args.val_data_dir, args.test_data_dir)
train_generator, validation_generator, test_generator = dataLoader.load_images()
checkpoint_1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
checkpoint_2 =  TensorBoard(log_dir='logs/{}', histogram_freq=0, write_graph=True, write_images=True)
callbacks_list = [checkpoint_2,checkpoint_1]



#X_train, Y_train, X_test, Y_test, val_data = dataLoader.load_images()
#print('Training data size', len(X_train))
start = time.time()
CNN.fit_generator(train_generator, steps_per_epoch =100, epochs = 100,callbacks=callbacks_list,validation_data  = validation_generator,validation_steps = 2 )
print('--------Test data--------')

x = CNN.evaluate_generator(test_generator, steps = 10, verbose = 1)
print(x)
end = time.time()
print('Time taken: ', str(end-start))
