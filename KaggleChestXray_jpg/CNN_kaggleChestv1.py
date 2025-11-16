# See https://www.kaggle.com/code/ghitabenjrinija/medical-image-analysis-with-cnn
# image data set from: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
#  Worked in TPU gcp,Colab T5 with some modifications

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.preprocessing import image
import numpy as np

from tensorflow.keras.optimizers import Adam


import datetime

train_dir = '/home/martinworkmbcet/.cache/kagglehub/datasets/paultimothymooney/chest-xray-pneumonia/versions/2/chest_xray/train'
test_dir = '/home/martinworkmbcet/.cache/kagglehub/datasets/paultimothymooney/chest-xray-pneumonia/versions/2/chest_xray/test'
val_dir = '/home/martinworkmbcet/.cache/kagglehub/datasets/paultimothymooney/chest-xray-pneumonia/versions/2/chest_xray/val'

# Image dimensions and batch size
image_size = (224, 224)
batch_size = 32

# Data augmentation for the training dataset
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Preprocess and augment the training data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

# Preprocess the test and validation data
test_datagen = ImageDataGenerator(rescale=1.0/255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)


val_generator = test_datagen.flow_from_directory(
    val_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)



base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

x = layers.Flatten()(base_model.output)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(base_model.input, x)



# Compile the model
model.compile(optimizer=Adam(0.0001) , loss='binary_crossentropy', metrics=['accuracy'])

print(datetime.datetime.now())
# Train the model
history = model.fit(train_generator, epochs=10, validation_data=val_generator)
print(datetime.datetime.now())

test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')


# Save the model
model.save('./cnn_model.h5')

# Test Prediction



# Load the trained model
model = tf.keras.models.load_model('./cnn_model.h5')

# Load an example image for prediction
image_path = '/home/martinworkmbcet/.cache/kagglehub/datasets/paultimothymooney/chest-xray-pneumonia/versions/2/chest_xray/val/PNEUMONIA/person1954_bacteria_4886.jpeg'
img = image.load_img(image_path, target_size=(224, 224))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)

# Make prediction
predictions = model.predict(img)

# Interpret the prediction
if predictions[0] < 0.5:
    print("The image is NORMAL. / value=",predictions[0])
else:
    print("The image indicates PNEUMONIA./value=",predictions[0])
