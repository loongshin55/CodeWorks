```python
import requests, json
from bs4 import BeautifulSoup as bs
from flask import *

app = Flask(__name__)
# 儲存股票數據
stock_data = {}

# 首頁
@app.route("/")
def index():
    return render_template("Lab12_plus.html")

# 獲取股票數據
@app.route('/stock', methods=['GET'])
def get_stock_data():
    number = request.args.get('name')  # 獲取股票代碼
    # 需檢查輸入資料是股票數字還是其他字母
    try:
        # 檢查輸入的是否是整數
        number_int = int(number)
        url = f'https://tw.stock.yahoo.com/quote/{number}.TW'
        r = requests.get(url)
        soup = bs(r.text, "html.parser")
        # 獲取股票名稱
        h1_outer = soup.find('h1', {'class': 'C($c-link-text) Fw(b) Fz(24px) Mend(8px)'})
        div_outer = soup.find('div', {'class': 'D(f) Ai(fe) Mb(4px)'})
        # 獲得股票價格，不過前面會有'生成'兩字，需要切片
        li_find = soup.find('li',{'class':'price-detail-item H(32px) Mx(16px) D(f) Jc(sb) Ai(c) Bxz(bb) Px(0px) Py(4px) Bdbs(s) Bdbc($bd-primary-divider) Bdbw(1px)'})
        # 切片
        li_new = li_find.text[2:]
        # 將名稱與價格加入字典
        stock_data.setdefault(h1_outer.text, li_new)
        # 將字典轉為json格式
        stock_data_string = json.dumps(stock_data, ensure_ascii = False)
        # 在後端輸出輸入的股票數字以及所截取到的資料
        print(f'user input data : {number}')
        print(f'Data on server : {stock_data}')
        # 返回資料給前端
        return {'output' : stock_data_string}
    # 若在是否為整數中，有報錯
    except Exception:
        # 若字串為'y'，則回傳到reset頁面且清除資料
        if number == 'y':
            return redirect(url_for('reset/y'))
        # 若為其他字串，則輸出原先的資料
        else:
            stock_data_string = json.dumps(stock_data, ensure_ascii = False)
            return {'output' : stock_data_string}

# Reset頁面
@app.route('/reset/<keyword>', methods=['GET'])
def reset(keyword):
    # 若keyword是'y'，則清除字典，且跳轉到有Reset！字樣以及超鏈接的頁面
    if keyword == 'y':
        stock_data.clear()
        # 在後端輸出被清空的字典
        print(f'Data on server : {stock_data}')
        return '''
            <h1>Reset!</h1>
            <p><a href="/">返回主頁</a></p>
        '''

app.run(host="0.0.0.0", port=4000, debug=True)
```
