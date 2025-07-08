<details>
  <summary>SVD.py</summary>

```python
from google.colab import drive
import sys
drive.mount('/content/drive/')
sys.path.append('/content/drive/MyDrive/線性代數作業/')

# _*_ coding:utf-8_*_
import numpy as np #導入numpy,且np為簡稱
import cv2 #導入cv2

# 使用絕對路徑或確認圖片檔案與notebook在同一目錄下
# 假設圖片檔案在 '/content/drive/MyDrive/線性代數作業/' 目錄下
img_org = cv2.imread('/content/drive/MyDrive/線性代數作業/test.jpg') #讀取圖片並儲存在img_org中

# 檢查圖片是否成功載入
if img_org is None:
    print("Error: Could not read the image file 'test.jpg'.")
    # 可以選擇在此處退出程式或進行其他錯誤處理
    exit() # 例如，直接退出

print('image shape is ', img_org.shape) #打印原本的圖片形狀,應顯示(高度, 寬度, 顏色通道數)

#定義SVD壓縮函數
def svd_compression(img, k): #定義一個函數,接受圖像img和奇異值數量k參數
  res_image = np.zeros_like(img, dtype=np.float64) #創建一個與IMG形狀相同的零矩陣來儲存壓縮後的結果, 使用浮點數類型
  for i in range(img.shape[2]): #迭代圖像的每個顏色通道(R,G,B)
    #進行奇異值分解,從svd函數得到的奇異值sigma是從大到小排列的
    U, Sigma, VT = np.linalg.svd(img[:,:,i]) #對第i個顏色通道進行SVD分解,得到U矩陣,奇異值Sigma和VT矩陣

    # 確保k值不超過奇異值的數量
    k_actual = min(k, Sigma.shape[0])

    # 重構壓縮後的圖像通道，選擇U的前k_actual列
    res_image[:,:,i] = U[:, :k_actual].dot(np.diag(Sigma[:k_actual])).dot(VT[:k_actual,:])


  return res_image

#保留前k個奇異值,這四行對圖像進行不同程度的壓縮,分別保留300,200,100和50個奇異值
res1 = svd_compression(img_org,k=544)
res2 = svd_compression(img_org,k=100)
res3 = svd_compression(img_org,k=50)
res4 = svd_compression(img_org,k=25)

row11 = np.hstack((res1, res2)) #將res1和res2水平拼接成一行
row22 = np.hstack((res3, res4)) #將res3和res4水平拼接成一行
res = np.vstack((row11,row22)) #將row11和row22垂直拼接成最終結果

from google.colab.patches import cv2_imshow # Import the necessary function

cv2_imshow(res) #顯示拼接後的圖像
# cv2.waitKey(0) #等待按鍵輸入以關閉顯示窗口
# cv2.destroyAllWindows() #關閉所有OpenCV顯示窗口
```

</details> 


<details>
  <summary>test.jpg & output</summary>

<h2>**test**<h2/>

![SVD原圖](https://github.com/user-attachments/assets/c09ed807-0b43-4ec1-8006-2c3915ae2e13)

<h2>**output**<h2/>

![SVD](https://github.com/user-attachments/assets/d858a1ae-e630-4057-95a9-c76fb921d412)


</details> 
