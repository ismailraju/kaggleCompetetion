# Convolutional Neural Network

# Installing Keras
# Enter the following command in a terminal (or anaconda prompt for Windows users): conda install -c conda-forge keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (28, 28, 1), activation = 'relu'))
#classifier.add(Conv2D(32, (3, 3), input_shape = (784, 1, 1), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 56, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

##################################################################################
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, 1: ].values
y = dataset.iloc[:, 0].values


#X_array = []
# 
#for x in X:
#     x[x<100]=0
#     x[x>100]=1
#   
#     X_array.append(np.reshape(x,(-1,28)))
#    
#X=X_array

c=np.ones((len(X),28,28,1));

for i in range( len(X)):
  print(i)
  c[i,:]=  np.reshape(X[i],(28,28,1)) 


X=c


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

 

#b=np.reshape(X_train[1],(-1,28))
 

#
#training_set = X_train
#
#test_set =X_test
#
#
#classifier.fit_generator((X_train,y_train),
#                         steps_per_epoch = 28000,
#                         epochs = 25,
#                         validation_data =(X_test, y_test),
#                         validation_steps = 14000)




from keras.preprocessing.image import ImageDataGenerator
# initialize the number of epochs and batch size
EPOCHS = 100
BS = 32

# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True, fill_mode="nearest")
 
 
training_set= aug.flow(X_train, y_train, batch_size=BS)
  
import matplotlib.pyplot as plt
p = training_set.next()
plt.imshow(p[0][0][:,:,0], cmap='gray')
plt.show()

# train the network
H = classifier.fit_generator(training_set,
	validation_data=(X_test, y_test), steps_per_epoch=len(X_train) // BS,
	epochs=EPOCHS)