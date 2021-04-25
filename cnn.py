import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

#Image Formatting
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale=1./255)

#Loading data
training_set = train_datagen.flow_from_directory(r'.\Data\train',
                                                 target_size = (64, 64), #Convert image to 64X64
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(r'.\Data\test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')
# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Adding a Convolution layer
cnn.add(tf.keras.layers.Conv2D(filters=40,padding="same",kernel_size=3, activation='relu', input_shape=[64, 64, 3])) #kernel_size is the filter shape, here it is 3X3. input_shape is 64x64X3 coz its rgb

# Adding a Pooling layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=40,padding='same',kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2)) #Adding another pooling layer

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=4, activation='softmax'))

#Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'CategoricalCrossentropy', metrics = ['accuracy'])
cnn.fit(x = training_set, validation_data = test_set, epochs = 20)

#Saving the model
cnn.save('modelcotton.h5')
print("Saved model to disk")

