```python
import matplotlib.pyplot as plt
import csv

from Tools.demo.sortvisu import steps
from matplotlib.ticker import MaxNLocator
from fontTools.merge.util import first


plt.rc('font', family='Microsoft JhengHei')
plt.style.use('bmh') # 設定風格
# 創建三個空集合，以儲存後面收集到數據
years = []
price = []
label = []

# 打開文件，並讀取，解碼方式需要使用big5(繁體中文文件)
file = open('price.csv', mode='r', newline='', encoding='big5')
r = csv.reader(file)
first_line = next(r)

# 將標題加入label集合裏
for i in range(1,len(first_line)):
    label.append(first_line[i])

# 處理數據
for row in r:
    # 將第一行的年份放入years集合
    years.append(row[0])
    # 將第一行之外的價格放入price集合
    price.append([float(price) for price in row[1:]])

# 折綫圖
plt.plot(years, price ,label=label)
# x軸數據的顯示間隔為5
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, steps=[5]))
plt.legend()
plt.xlabel('時間') # 標注x軸
plt.ylabel('價格') # 標注y軸
plt.show()
```
