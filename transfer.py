import pandas as pd

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input

batch_size = 30
epochs = 10
IMG_HEIGHT = 256
IMG_WIDTH = 256
train_dir = "prep-spectrograms-torento"

CLASS_NAMES = ['neutral', 'fear', 'happy', 'disgust', 'angry', 'sad', 'ps']

data = pd.read_csv('/content/drive/My Drive/all_data_torento.csv', names=['path', 'cls'])

train_data_gen = ImageDataGenerator().flow_from_dataframe(
                dataframe=data[:int(data.shape[0]*.7)],
                y_col='cls', x_col='path',batch_size=batch_size,directory=train_dir,
                shuffle=True, target_size=(IMG_HEIGHT, IMG_WIDTH),
                classes=CLASS_NAMES)

valid_data_gen = ImageDataGenerator().flow_from_dataframe(
                dataframe=data[int(data.shape[0]*.7):int(data.shape[0]*.9)],
                y_col='cls', x_col='path', batch_size=batch_size,directory=train_dir,
                shuffle=True, target_size=(IMG_HEIGHT, IMG_WIDTH),
                classes=CLASS_NAMES)

test_data_gen = ImageDataGenerator().flow_from_dataframe(
                dataframe=data[int(data.shape[0]*.9):],
                y_col='cls', x_col='path', batch_size=batch_size,directory=train_dir,
                shuffle=True, target_size=(IMG_HEIGHT, IMG_WIDTH),
                classes=CLASS_NAMES)  


img_input = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
vgg19_model = VGG16(input_tensor=img_input,weights="imagenet")
o = vgg19_model.get_layer('block4_pool').output
o = Flatten()(o)
o = Dense(1024, activation='relu')(o)
output = Dense(7, activation='softmax')(o)
model = Model(inputs=img_input, outputs=output)


checkpoint = ModelCheckpoint('weights_cnn_torento_vgg.hdf5', monitor='val_acc', 
                verbose=1, save_best_only=True,mode='auto')

es = EarlyStopping(mode='min', monitor='val_loss', min_delta=0.001, verbose=1)

adam = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

print("Traning Model...")
history = model.fit_generator(train_data_gen, epochs=epochs, verbose=1,
                        validation_data=valid_data_gen, callbacks=[checkpoint, es],
                        steps_per_epoch=len(train_data_gen), validation_steps=len(valid_data_gen))


# test
y_pred = model.predict(test_data_gen) 
y_pred_labels = np.argmax(y_pred, axis=1)

print('VGG16,acc on test:\t', accuracy_score(test_data_gen.classes, y_pred_labels))
confusion_matrix = metrics.confusion_matrix(y_true=test_data_gen.classes, y_pred=y_pred_labels)

print(confusion_matrix)

plt.style.use("ggplot")
fig = plt.figure(figsize=(20,8))


fig.add_subplot(1,2,1)
plt.title("Training Accuracy on VGG16")
plt.plot(history.history["acc"], label="train_accuracy")
plt.plot(history.history["val_acc"], label="val_accuracy")
plt.ylim(0, 1)

plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")
plt.show()