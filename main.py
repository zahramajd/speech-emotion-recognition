#TODO
# confusion matrix
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Conv2D, MaxPooling2D, Flatten, Dropout, Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model


batch_size = 30
epochs = 30
IMG_HEIGHT = 256
IMG_WIDTH = 256
train_dir = "prep-spectrograms"

CLASS_NAMES = ['T', 'N', 'E', 'L', 'F', 'W', 'A']

train_image_generator = ImageDataGenerator()
val_image_generator = ImageDataGenerator()

data = pd.read_csv('all_data.csv', names=['path', 'cls'])

train_data_gen = train_image_generator.flow_from_dataframe(
                dataframe=data[:int(data.shape[0]*.9)],
                y_col='cls', x_col='path',batch_size=batch_size,directory=train_dir,
                shuffle=True, target_size=(IMG_HEIGHT, IMG_WIDTH),
                classes=CLASS_NAMES)

valid_data_gen = val_image_generator.flow_from_dataframe(
                dataframe=data[int(data.shape[0]*.9):],
                y_col='cls', x_col='path', batch_size=batch_size,directory=train_dir,
                shuffle=True, target_size=(IMG_HEIGHT, IMG_WIDTH),
                classes=CLASS_NAMES)


model = Sequential()
model.add(Conv2D(120, kernel_size=(11, 11), strides=(4,4), padding='valid',
            kernel_initializer='normal', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)))
model.add(MaxPooling2D((3, 3), strides=(2, 2)))

model.add(Conv2D(256, kernel_size=(5, 5), strides=(1,1), padding='valid',
                    kernel_initializer='normal', activation='relu'))

model.add(Conv2D(120, kernel_size=(11, 11), strides=(4,4), padding='valid',
                    kernel_initializer='normal', activation='relu'))

model.add(GlobalAveragePooling2D())
model.add(Dense(2048, activation='relu', name='fc1'))
model.add(Dropout(0.5))
model.add(Dense(2048, activation='relu', name='fc2'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax', name='fc3'))

checkpoint = ModelCheckpoint('weights_cnn.hdf5', monitor='val_accuracy', 
                verbose=1, save_best_only=True,mode='auto')

tb_plot = TensorBoard(log_dir='.', histogram_freq=0, write_graph=True, write_images=True)

adam = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

print("Traning Model...")
history = model.fit_generator(train_data_gen, epochs=epochs, verbose=1,
                        callbacks=[checkpoint, tb_plot], validation_data=valid_data_gen, 
                        steps_per_epoch=len(train_data_gen), validation_steps=len(valid_data_gen))

