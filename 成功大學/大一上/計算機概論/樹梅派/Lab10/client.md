```python
from socket import socket, SOCK_STREAM, AF_INET

client = socket(family=AF_INET, type=SOCK_STREAM)
# 2. 連接到伺服器(需要指定IP位址和port)
client.connect(('127.0.0.1', 5678))
print('連線成功')
# 3. 從伺服器接收資料
msg = input('請輸入訊息:\n')
client.send(msg.encode('utf-8'))
# print(client.recv(1024).decode('utf-8'))
client.close()
# ssh handsome@192.168.137.37
```
