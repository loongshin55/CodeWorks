```python
#coding=utf-8
# 不加上就無法使用中文註解

import random
from flask import *
app = Flask(__name__)

# 使用render_template函式讓首頁的路由能夠回傳html檔(簡單架構)給前端，來顯示首頁畫面，而不是僅用return文字的方式
@app.route("/")
def index():
    # 回傳首頁畫面 ( 請確保你的html檔是放在名為 templates的資料夾) 
    return render_template('Lab11_plus.html')


@app.route('/sort_numbers', methods = ['POST'])
def sort_numbers():
    # 創立一個空列表，為等下儲存數字
    number_list = []
    # 以記錄步驟的空列表
    steps = []
    # 從網頁上截取數字
    number = request.form['numbers']
    # 將數字以，分開
    numbers = number.split(',')
    # 將數字都轉換爲int類型以及放列表當中
    for i in numbers:
        i = int(i)
        number_list.append(i)
    # 在terminal端顯示數字排序前的順序
    print(f'排序前 : {number_list}')
    # 使用插入排序法
    # 從前兩個數字先開始比較，若第一個數字較大，則兩個數字交換；若第二個數字較大，則保持位置
    # 再使用第三個數字跟第二個比較，若第二個數字較大，則交換位置；若第三個數字較大，則保持位置
    # 在第二個數字較大的情況下，第三個數字須再與第一個數字比較，若第一個數字較大，再交換位置；若第三個數字較大，則保持位置
    # 依照這種流程，排序完五個數字
    for i in range(1, len(number_list)):
        base = number_list[i]
        j = i - 1
        while j >= 0 and number_list[j] > base:
            number_list[j + 1] = number_list[j]
            j -= 1
        number_list[j + 1] = base
        # 將每一步的步驟存入steps當中
        steps.append(list(number_list))
    # 在terminal端顯示排序後的數字順序
    print(f'排序後 : {number_list}')
    # 創立result，然後將要回傳給網頁的文字，存入result當中，先存入初始列表的部分
    result = f'初始列表: {number}<br>'
    # 使用for循環，將步驟存入result當中
    # enumerate可以同時返回内容以及對應的索引值
    # enumerate(對象，start=想要開始的索引值)
    for index, step in enumerate(steps, start=1):
        result += f'第{index}步: {step}<br>'
    # 最後將最終結果存入result
    result += f'最終結果:{number_list}'
    # 回傳result到網頁中顯示
    return result

# 生成隨機數組
@app.route('/generate_random',methods = ['GET'])
def generate_random_numbers():
    # 創立一個儲存數字的空列表
    random_numbers = []
    # 生成五個0到100之間的隨機數組，并將數組儲存在列表當中
    for i in range(5):
        # 生成整數數組，且random(a,b)是包括a以及b
        random_int = random.randint(0,100)
        random_numbers.append(random_int)
    # 在terminal端顯示隨機生成的數組
    print(random_numbers)
    # 將數組回傳到網頁上
    return {'numbers': random_numbers}
    


app.run(host="0.0.0.0", port=4000, debug=True)
```
