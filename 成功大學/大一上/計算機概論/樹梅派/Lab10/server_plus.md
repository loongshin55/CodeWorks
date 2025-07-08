```python
from socket import socket, SOCK_STREAM, AF_INET
from threading import Thread


def handle_client(client, addr):
    # 處理與客戶端的互動
    print(f'{addr} 連接到了伺服器。')

    while True:
        # 持續接收來自客戶端的訊息
        msg = client.recv(1024).decode('utf-8')
        if not msg:  # 客戶端斷開連接
            print(f'{addr} 的連接已關閉。')
            break

        print(f"接收到來自 {addr} 的訊息: {msg}")

        # 如果接收到 'goodbye' 訊息則關閉連接
        if msg.lower() == 'goodbye':
            print(f'{addr} 發送了 goodbye，關閉連接。')
            client.send("Goodbye, closing connection.".encode('utf-8'))  # 可選，發送回應
            break

    client.close()
    print(f'{addr} 的連接已關閉。')


# 1. 創建Socket物件並指定使用哪種IP格式、傳輸協議
# family=AF_INET - IPv4位址
# family=AF_INET6 - IPv6位址
# type=SOCK_STREAM - TCP套接字
# type=SOCK_DGRAM - UDP套接字
server = socket(family=AF_INET, type=SOCK_STREAM)
# 2. 綁定IP位址和port(port用於區分不同的服務)
server.bind(('127.0.0.1', 5678))
# 3. 開啟監聽 - 監聽客戶端連接到伺服器
# 參數來限制最多能有幾個客戶端排隊連接伺服器
server.listen(5)
print('伺服器啟動，開始監聽...')

while True:
    # 4. 通過迴圈接收客戶端的連接並作出相應的處理(提供服務)
    # accept方法是一個阻塞方法，如果沒有客戶端連接到伺服器，程式碼不會繼續向下執行
    # accept方法返回一個元組，其中的第一個元素是客戶端物件
    # 第二個元素是連接到伺服器的客戶端地址(由IP和port兩部分構成)
    client, addr = server.accept()
    # 5. 為每個客戶端新建一個執行緒
    client_thread = Thread(target=handle_client, args=(client, addr))
    client_thread.start()
```
