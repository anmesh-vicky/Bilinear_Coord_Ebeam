from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras import metrics

class base_model:
    def __init__(self,learning_rate,dropout,kernel_size):
        self.learning_rate = learning_rate
#         self.epochs = epochs
#         self.batch_size = batch_size
        #self.activations = activations
        self.dropout = dropout
        self.kernel_size = kernel_size
# # create the base pre-trained model
   
    
    def build_model(self):
        base_model = InceptionV3(weights='imagenet', include_top=False)
# add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
		# and a logistic layer -- let's say we have 200 classes
        predictions = Dense(5, activation='softmax')(x)

		# this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)

		# first: train only the top layers (which were randomly initialized)
		# we should freeze:
#         for i, layer in enumerate(base_model.layers):
#             print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
#         for layer in model.layers[:249]:
#             layer.trainable = False
#         for layer in model.layers[249:]:
#             layer.trainable = True

        model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics = ['accuracy'])
        return model

		# train the model on the new data for a few epochs
		# model.fit_generator(...)

		# # at this point, the top layers are well trained and we can start fine-tuning
		# # convolutional layers from inception V3. We will freeze the bottom N layers
		# # and train the remaining top layers.

		# # let's visualize layer names and layer indices to see how many layers
		# # we should freeze:
		# for i, layer in enumerate(base_model.layers):
		#    print(i, layer.name)

		# # we chose to train the top 2 inception blocks, i.e. we will freeze
		# # the first 249 layers and unfreeze the rest:
		# for layer in model.layers[:249]:
		#    layer.trainable = False
		# for layer in model.layers[249:]:
		#    layer.trainable = True

		# # we need to recompile the model for these modifications to take effect
		# # we use SGD with a low learning rate
		# from keras.optimizers import SGD
		# model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# # we train our model again (this time fine-tuning the top 2 inception blocks
# # alongside the top Dense layers
# model.fit_generator(...)
