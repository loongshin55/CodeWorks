### colab環境
<details>
  <summary>CNN.py</summary>


  ```python
  from google.colab import drive
import sys
drive.mount('/content/drive/')
sys.path.append('/content/drive/MyDrive/線性代數作業/')

import matplotlib.pyplot as plt

def plot_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2, 2)
    plt.imshow(image, cmap='binary')
    plt.show()


def plot_images_labels_predict(images, labels, prediction, idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 25: num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1 + i)
        ax.imshow(images[idx], cmap='binary')
        title = "l=" + str(labels[idx])
        if len(prediction) > 0:
            title = "l={},p={}".format(str(labels[idx]), str(prediction[idx]))
        else:
            title = "l={}".format(str(labels[idx]))
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]);
        ax.set_yticks([])
        idx += 1
    plt.show()



def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

###import lib_plot
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D


#---------------preprocess------------------
np.random.seed(10)

# Read MNIST data
(X_Train, y_Train), (X_Test, y_Test) = mnist.load_data()

X_Train_norm = X_Train/255
X_Test_norm = X_Test/255


#---------build cnn-----------
model = Sequential()

# Create Convolution layer
model.add(Conv2D(filters=32,#填入程式碼
                 kernel_size=(5,5 ),#填入程式碼
                 padding='same',
                 input_shape=(28, 28, 1),
                 activation='relu',
                 name='conv2d_1'))

# Create Max-Pool
model.add(MaxPool2D(pool_size=(2,2 ), name='max_pooling2d_1'))#填入程式碼

# Add hidden layer
model.add(Dropout(0.25, name='dropout_1'))
model.add(Flatten(name='flatten_1'))
model.add(Dense(10, activation='softmax', name='dense_2'))

# Show model summary
model.summary()
print("")


#---------------train model------------------
# # Compile the model with loss function, optimizer, and evaluation metric
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Start training model
train_history = model.fit(x=X_Train_norm,
                          y=y_Train, validation_split=0.2,
                          epochs=10, batch_size=300, verbose=1)

# Plot train history
show_train_history(train_history, 'accuracy', 'val_accuracy')
show_train_history(train_history, 'loss', 'val_loss')

###lib_plot.show_train_history(train_history, 'accuracy', 'val_accuracy')
###lib_plot.show_train_history(train_history, 'loss', 'val_loss')


#-------------------evaluate-----------------
# Evaluate the model using the test data and get the loss and accuracy
scores = model.evaluate(X_Test_norm, y_Test)

# Print testing accuracy
print()
print("Train accuracy = {:.2f}%".format(train_history.history['accuracy'][-1] * 100))
print("Validation accuracy = {:.2f}%".format(train_history.history['val_accuracy'][-1] * 100))
print("Test accuracy = {:.2f}%".format(scores[1] * 100))


# Use the trained model to make predictions on the test data
predictions = model.predict(X_Test_norm)

prediction_classes = np.argmax(predictions, axis=1)
plot_images_labels_predict(X_Test, y_Test, prediction_classes, idx=240)

  ```

<h2> output <h2/>

  
![CNN線圖9855%](https://github.com/user-attachments/assets/5d13211d-dabb-4f22-b6eb-f1bb9f8a4f6d)
![image](https://github.com/user-attachments/assets/cf0d5e83-5900-40a6-b635-2b8f91848fd3)



</details> 


<details>
  <summary>CNN-ViT.py</summary>


  ```python
  from google.colab import drive
import sys
drive.mount('/content/drive/')
sys.path.append('/content/drive/MyDrive/線性代數作業/')


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# ---------- 資料處理 ----------
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 將輸入轉換為 RGB（ViT 要求 3 通道）
x_train = tf.image.grayscale_to_rgb(tf.expand_dims(x_train, -1))
x_test = tf.image.grayscale_to_rgb(tf.expand_dims(x_test, -1))

# 調整尺寸為 32x32（ViT 通常使用大一點輸入）
x_train = tf.image.resize(x_train, [32, 32])
x_test = tf.image.resize(x_test, [32, 32])

# ---------- 建立 Vision Transformer ----------
# Patch 分割
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

# Patch 編碼
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

# 建立 ViT 模型
def create_vit_model():
    input_shape = (32, 32, 3)
    patch_size = 4
    num_patches = (32 // patch_size) ** 2
    projection_dim = 64
    num_heads = 4
    transformer_units = [128, 64]
    transformer_layers = 4
    mlp_head_units = [128, 64]

    inputs = layers.Input(shape=input_shape)
    patches = Patches(patch_size)(inputs)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim)(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = keras.Sequential([
            layers.Dense(units=transformer_units[0], activation=tf.nn.gelu),
            layers.Dense(units=transformer_units[1])
        ])(x3)
        encoded_patches = layers.Add()([x3, x2])

    x = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    for units in mlp_head_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(0.3)(x)
    logits = layers.Dense(10, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=logits)
    return model

# 建立與編譯模型
model = create_vit_model()
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# ---------- 模型訓練 ----------
history = model.fit(x_train, y_train,
                    validation_split=0.2,
                    epochs=10,
                    batch_size=128,
                    verbose=1)

# ---------- 畫出訓練歷程 ----------
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

show_train_history(history, 'accuracy', 'val_accuracy')
show_train_history(history, 'loss', 'val_loss')

# ---------- 評估與預測 ----------
scores = model.evaluate(x_test, y_test, verbose=0)
print()
print("Train Accuracy: {:.2f}%".format(history.history['accuracy'][-1]*100))
print("Validation Accuracy: {:.2f}%".format(history.history['val_accuracy'][-1]*100))
print("Test Accuracy: {:.2f}%".format(scores[1]*100))

predictions = model.predict(x_test)
y_pred_classes = np.argmax(predictions, axis=1)

# ---------- 顯示預測圖 ----------
def plot_images_labels_predict(images, labels, prediction, idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 25: num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1 + i)
        ax.imshow(images[idx], cmap='binary')
        title = "l={}".format(str(labels[idx]))
        if len(prediction) > 0:
            title = "l={},p={}".format(str(labels[idx]), str(prediction[idx]))
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        idx += 1
    plt.show()

plot_images_labels_predict(x_test.numpy().squeeze(), y_test, y_pred_classes, idx=240)

  ```

<h2> output <h2/>

  
![ViT線圖](https://github.com/user-attachments/assets/fa1f72ee-9318-42a2-8367-2ad0937c4b4d)
![image](https://github.com/user-attachments/assets/94707758-d6ac-45e9-8c29-605b9775a2ba)


</details> 


