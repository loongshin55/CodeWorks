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
