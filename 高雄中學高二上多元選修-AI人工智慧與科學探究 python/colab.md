
```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score

#-------------視覺化工具-------------
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
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        idx += 1
    plt.show()

#-------------Hebbian 函數-------------
def hardlim(x):
    return np.where(x >= 0, 1, -1)

def recall(W, x, steps=5):
    for _ in range(steps):
        x = hardlim(np.dot(W, x))
    return x

def predict_hebbian(W, prototypes, labels, x_test_bin):
    preds = []
    recalls = []
    for x in x_test_bin:
        x_recalled = recall(W, x)
        sims = [np.dot(x_recalled, p) for p in prototypes]
        pred_label = labels[np.argmax(sims)]
        preds.append(pred_label)
        recalls.append(x_recalled)
    return np.array(preds), np.array(recalls)

#-------------載入資料-------------
(X_Train, y_Train), (X_Test, y_Test) = mnist.load_data()

#-------------固定類別-------------
selected_digits = [0, 1, 2]

#-------------資料篩選與處理-------------
train_mask = np.isin(y_Train, selected_digits)
test_mask = np.isin(y_Test, selected_digits)

X_train_sel = X_Train[train_mask]
y_train_sel = y_Train[train_mask]
X_test_sel = X_Test[test_mask]
y_test_sel = y_Test[test_mask]

X_train_bin = np.where(X_train_sel > 127, 1, -1).reshape(X_train_sel.shape[0], -1)
X_test_bin = np.where(X_test_sel > 127, 1, -1).reshape(X_test_sel.shape[0], -1)

#-------------實驗：準確率 vs 樣本數-------------
sample_counts = list(range(1, 201, 10))  # 從 1 到 200，每類遞增 10 個樣本
accuracies = []

for samples_per_class in sample_counts:
    prototypes = []
    labels = []

    for digit in selected_digits:
        class_data = X_train_bin[y_train_sel == digit][:samples_per_class]
        avg_pattern = np.mean(class_data, axis=0)
        avg_pattern = np.sign(avg_pattern)
        prototypes.append(avg_pattern)
        labels.append(digit)

    prototypes = np.array(prototypes)

    # 建立 Hebbian 權重
    W = np.zeros((784, 784))
    for p in prototypes:
        W += np.outer(p, p)
    np.fill_diagonal(W, 0)

    # 預測
    y_pred, _ = predict_hebbian(W, prototypes, labels, X_test_bin)
    acc = accuracy_score(y_test_sel, y_pred)
    accuracies.append(acc)

#-------------畫圖-------------
plt.figure(figsize=(10, 6))
plt.plot(sample_counts, np.array(accuracies) * 100, marker='o')
plt.title("Hebbian Test Accuracy vs Training Samples per Class")
plt.xlabel("Samples per Class")
plt.ylabel("Test Accuracy (%)")
plt.grid(True)
plt.xticks(sample_counts)
plt.show()


import numpy as np
def hardlim(x):
    if x<0:
        x=0
    elif x>=0:
        x=1
    return x

p0=np.array([
            [-1, 1, 1, 1,-1],
            [ 1,-1,-1,-1, 1],
            [ 1,-1,-1,-1, 1],
            [ 1,-1,-1,-1, 1],
            [ 1,-1,-1,-1, 1],
            [-1, 1, 1, 1,-1]
            ])
p1=np.array([
            [-1,-1, 1,-1,-1],
            [-1,-1, 1,-1,-1],
            [-1,-1, 1,-1,-1],
            [-1,-1, 1,-1,-1],
            [-1,-1, 1,-1,-1],
            [-1, 1, 1,-1,-1]
            ])
p2=np.array([
            [-1, 1, 1, 1, 1],
            [-1, 1,-1,-1,-1],
            [-1, 1, 1,-1,-1],
            [-1,-1,-1, 1,-1],
            [-1,-1,-1, 1,-1],
            [ 1, 1, 1,-1,-1]
            ])
p0t1=np.array([
            [-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1],
            [ 1,-1,-1,-1, 1],
            [ 1,-1,-1,-1, 1],
            [-1, 1, 1, 1,-1]
            ])
p1t1=np.array([
            [-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1],
            [-1,-1, 1,-1,-1],
            [-1,-1, 1,-1,-1],
            [-1, 1, 1,-1,-1]
            ])
p2t1=np.array([
            [-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1],
            [-1,-1,-1, 1,-1],
            [-1,-1,-1, 1,-1],
            [ 1, 1, 1,-1,-1]
            ])
p0t2=np.array([
            [-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1],
            [ 1,-1,-1,-1, 1],
            [-1, 1, 1, 1,-1]
            ])
p1t2=np.array([
            [-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1],
            [-1,-1, 1,-1,-1],
            [-1, 1, 1,-1,-1]
            ])
p2t2=np.array([
            [-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1],
            [-1,-1,-1, 1,-1],
            [ 1, 1, 1,-1,-1]
            ])
p0t3=np.array([
            [-1, 1,-1, 1,-1],
            [ 1,-1, 1,-1, 1],
            [-1,-1,-1,-1,-1],
            [ 1,-1,-1,-1, 1],
            [ 1,-1,-1, 1,-1],
            [-1,-1, 1, 1,-1]
            ])
p1t3=np.array([
            [ 1, 1, 1, 1,-1],
            [-1,-1, 1,-1,-1],
            [-1,-1,-1,-1, 1],
            [ 1,-1, 1,-1,-1],
            [-1,-1, 1, 1,-1],
            [-1, 1, 1,-1,-1]
            ])
p2t3=np.array([
            [ 1, 1, 1,-1, 1],
            [-1,-1,-1,-1,-1],
            [-1, 1, 1,-1, 1],
            [-1,-1,-1,-1,-1],
            [-1, 1,-1, 1,-1],
            [-1, 1, 1,-1,-1]
            ])

import matplotlib.pyplot as plt

fig, axes = plt.subplots()
axes.pcolor(p0)

fig, axes = plt.subplots()
axes.pcolor(p1)

fig, axes = plt.subplots()
axes.pcolor(p2)

fig, axes = plt.subplots()
axes.pcolor(p0t1)

fig, axes = plt.subplots()
axes.pcolor(p1t1)

fig, axes = plt.subplots()
axes.pcolor(p2t1)

fig, axes = plt.subplots()
axes.pcolor(p0t2)

fig, axes = plt.subplots()
axes.pcolor(p1t2)

fig, axes = plt.subplots()
axes.pcolor(p2t2)

fig, axes = plt.subplots()
axes.pcolor(p0t3)

fig, axes = plt.subplots()
axes.pcolor(p1t3)

fig, axes = plt.subplots()
axes.pcolor(p2t3)

Pp1=np.zeros((3,30))
p_list=[p0,p1,p2]
i=0
for p in p_list:
  Pp1[i]=np.reshape(p,30)
  i=i+1


w2=np.matmul(np.linalg.pinv(Pp1),Pp1) #使用內建偽逆

w1=np.matmul(np.matmul(np.linalg.inv(np.matmul(Pp1,Pp1.T)),Pp1).T,Pp1) #使用理論偽逆 P+=(((P^T)P)^-1)P^T


p_list=[p0,p1,p2]
w0=np.zeros((30,30)) #製造權重空間
for p in p_list:
    p=np.reshape(p,30)#是將2d array TO 1D ARRAY
    p=p/np.linalg.norm(p) #取歸一
    w0+=np.outer(p,p) #取製作初始權重

t11=p2t3 #pxtx也就是題目
fig, axes = plt.subplots()
axes.pcolor(t11) #畫題目

fig, axes = plt.subplots()
t=np.reshape(t11,30) #是將2d array TO 1D ARRAY

t=t/np.linalg.norm(t11)  #取歸一
a=np.matmul(w0,t)
t=[]
for i in a:
  t.append(hardlim(i))
t=np.reshape(t,(6,5))
axes.pcolor(t)
