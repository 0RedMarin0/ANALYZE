import socket


def start_test_server():
    host = '0.0.0.0'  # Слушать все интерфейсы
    port = 12300

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        print(f"Сервер слушает на порту {port}...")

        conn, addr = s.accept()
        with conn:
            print(f"Подключился клиент: {addr}")
            data = conn.recv(1024)
            print(f"Получено: {data.decode()}")
            conn.sendall(b"Hello from server!")


if __name__ == "__main__":
    start_test_server()

# import socket
#
#
# def test_connection():
#     server_ip = '192.168.1.100'  # Замени на IP компьютера
#     port = 12300
#
#     try:
#         with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#             s.connect((server_ip, port))
#             s.sendall(b"Hello from client!")
#             data = s.recv(1024)
#             print(f"Ответ сервера: {data.decode()}")
#             print("Соединение успешно!")
#     except Exception as e:
#         print(f"Ошибка подключения: {e}")
#
#
# if __name__ == "__main__":
#     test_connection()