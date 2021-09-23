import os
import numpy as np
import tensorflow as tf
import pandas as pd
from matplotlib import pyplot
from sklearn.model_selection import train_test_split

### Load data set
df = pd.read_csv('dataset.csv')

### print first values
# print(df.head())

### print emotions
# print(df.emotion.unique())

label_to_text = {0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Happiness', 4: 'Sadness', 5: 'Surprise', 6: 'Neutral'}

### map de dataset
pyplot.imshow(np.array(df.pixels.loc[0].split(' ')).reshape(48, 48).astype('float'))

### Show the image
# pyplot.show()

fig = pyplot.figure(1, (14, 14))
k = 0

### classify images according to emotion
for label in sorted(df.emotion.unique()):
    for j in range(3):
        px = df[df.emotion == label].pixels.iloc[k]
        px = np.array(px.split(' ')).reshape(48, 48).astype('float32')
        k += 1
        ax = pyplot.subplot(7, 7, k)
        ax.imshow(px)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(label_to_text[label])
        pyplot.tight_layout()
# pyplot.show()

### do all the process to each image
img_array = df.pixels.apply(lambda x: np.array(x.split(' ')).reshape(48, 48, 1).astype('float32'))
img_array = np.stack(img_array, axis=0)
labels = df.emotion.values

### training
x_train, x_test, y_train, y_test = train_test_split(img_array, labels, test_size=0.2)

# print(x_train.shape, y_train.shape, x_test.shape,  y_test.shape )

##preparing the model
x_train = x_train / 255
x_test = x_test / 255

### declaring tensorflow model
### defining layer: kernel size, strides
### inputshape = 48, 48, 1
# output shape
###normaliza the data each iteration
basemodel = tf.keras.models.Sequential([tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
                                        tf.keras.layers.MaxPool2D(2, 2),
                                        tf.keras.layers.BatchNormalization(),
                                        # layer 2
                                        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
                                        tf.keras.layers.MaxPool2D(2, 2),
                                        tf.keras.layers.BatchNormalization(),
                                        # layer 3
                                        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(48, 48, 1)),
                                        tf.keras.layers.MaxPool2D(2, 2),
                                        tf.keras.layers.BatchNormalization(),
                                        # Classification layer
                                        tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dense(128, activation='relu'),
                                        tf.keras.layers.Dense(7, activation='softmax')

                                        ])

### check model
# print(basemodel.summary())

basemodel.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
try:
    os.mkdir('test')
except:
    pass

file_name = 'Saiko_moderu.h'

checkpoint_path = os.path.join('test', file_name)

call_back = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                               monitor='val_accuracy',
                                               verbose=1,
                                               save_freq='epoch',
                                               save_best_only=True,
                                               save_weights_only=False,
                                               mode='max')

basemodel.fit(x_train, y_train, epochs=20, validation_split=.1, callbacks=call_back)


