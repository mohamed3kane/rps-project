import socket
from PIL import Image
import io
import os

PORT = 5001
os.makedirs(os.path.expanduser("~/Desktop/realtime"), exist_ok=True)

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("0.0.0.0", PORT))
server.listen(1)
print("Waiting for ESP...")

conn, addr = server.accept()
print("ESP connected!")

frame = 0
while True:
    # Read label
    label_len = int.from_bytes(conn.recv(1), 'big')
    label = conn.recv(label_len).decode()

    # Read image
    raw_size = b""
    while len(raw_size) < 4:
        raw_size += conn.recv(4 - len(raw_size))
    size = int.from_bytes(raw_size, 'big')

    data = b""
    while len(data) < size:
        chunk = conn.recv(size - len(data))
        if not chunk:
            break
        data += chunk

    # Save and display
    img = Image.open(io.BytesIO(data))
    img_path = os.path.expanduser(f"~/Desktop/realtime/frame_{frame:04d}.bmp")
    img.save(img_path)
    img.show()

    print(f"Frame {frame}: {label}")
    frame += 1