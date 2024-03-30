import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import load_img
from keras.applications.imagenet_utils import preprocess_input
from sklearn import preprocessing
import os

def images_and_classes(root_dir):
    images_path = []
    classes = []

    for class_dir in os.listdir(root_dir):
        class_dir_path = os.path.join(root_dir, class_dir)

        # Check if the item in the dataset folder is a directory
        if os.path.isdir(class_dir_path):
            for image in os.listdir(class_dir_path):
                image_path = os.path.join(class_dir_path, image)

                # im = cv.imread(image_path)
                images_path.append(image_path)
                classes.append(class_dir)

                # with Image.open(image_path) as image:
                #     images.append(image)

    return images_path, classes


root_dir = 'blood cells/bloodcells_dataset/'

images_path, classes = (images_and_classes(root_dir))



images = [load_img(img, target_size=(224,224)) for img in images_path]

images = np.array(images)
images = preprocess_input(images)

# Count the number of samples per class
class_counts = {}

for label in classes:
    class_counts[label] = class_counts.get(label, 0) + 1


x = list(class_counts.keys())
y = list(class_counts.values())

plt.figure(figsize=(10,6))
plt.bar(x, y)
plt.xlabel('Class')
plt.ylabel('Number of samples')
plt.title('Number of samples per class')
plt.xticks(rotation=45, ha='right')
plt.show()


le = preprocessing.LabelEncoder()
classes = le.fit_transform(classes)


vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))

for layer in vgg.layers:
    layer.trainable = False

x = Flatten()(vgg.output)
predictions = Dense(8, activation='softmax')(x)

model = Model(inputs=vgg.inputs, outputs=predictions)

model.summary()


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Adjust learning rate as needed
              loss='sparse_categorical_crossentropy',  # Use appropriate loss function
              metrics=['accuracy']) 

history = model.fit(images, classes, epochs=10, batch_size=32)


## uncomment below lines to save the model weights
# final_weights_filepath = 'final_model_weights.h5'       # Save model weights to a file
# model.save_weights(final_weights_filepath)

plt.plot(history.history['accuracy'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Plot')
plt.legend(['Accuracy'])
plt.show()

plt.plot(history.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Plot')
plt.legend(['Loss'])
plt.show()