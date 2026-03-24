import socket
from PIL import Image
import io
import os

# --- Settings ---
LABEL = "paper"        # Change to "paper" or "scissors"
MAX_IMAGES = 500      # Stop after this many images
PORT = 5001

# Create folder for this label
save_dir = os.path.expanduser(f"~/Desktop/dataset/{LABEL}")  # same folder as before
os.makedirs(save_dir, exist_ok=True)

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("0.0.0.0", PORT))
server.listen(1)
print(f"Collecting {MAX_IMAGES} images for class: {LABEL}")
print("Waiting for ESP to connect...")

conn, addr = server.accept()
print("ESP connected! Starting capture...")

# Start numbering from existing count to avoid overwriting
existing = len([f for f in os.listdir(save_dir) if f.endswith('.bmp')])
frame_count = existing
print(f"Starting from image {frame_count}")

while frame_count < MAX_IMAGES:
    # Read image size
    raw_size = b""
    while len(raw_size) < 4:
        raw_size += conn.recv(4 - len(raw_size))
    size = int.from_bytes(raw_size, "big")

    # Read image data
    data = b""
    while len(data) < size:
        chunk = conn.recv(size - len(data))
        if not chunk:
            break
        data += chunk

    # Save image
    filename = f"{save_dir}/{LABEL}_{frame_count:04d}.bmp"
    with open(filename, "wb") as f:
        f.write(data)

    frame_count += 1
    print(f"Saved {frame_count}/{MAX_IMAGES}: {filename}")

print(f"\nDone! {MAX_IMAGES} images saved to {save_dir}/")
conn.close()
server.close()