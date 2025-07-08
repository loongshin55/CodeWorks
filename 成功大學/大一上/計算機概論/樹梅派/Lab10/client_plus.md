```python
from socket import socket, SOCK_STREAM, AF_INET

client = socket(family=AF_INET, type=SOCK_STREAM)
# 2. 連接到伺服器(需要指定IP位址和port)
client.connect(('127.0.0.1', 5678))
print('連線成功')

while True:
    # 3. 從使用者獲取訊息
    msg = input('請輸入訊息:\n')
    client.send(msg.encode('utf-8'))

    # 如果使用者輸入 'goodbye'，關閉連線
    if msg.lower() == 'goodbye':
        print('正在關閉連線...')
        break

# 客戶端發送完訊息後關閉連接
client.close()
```
