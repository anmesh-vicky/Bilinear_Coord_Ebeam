from keras.models import load_model
import model
#from keras.models import summary

model1 = model.model(0.1, 10, 32, 0.2, kernel_size = 5)
print(model1.summary())