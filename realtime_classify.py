from image_preprocessing import resize_96x96_to_32x32_and_threshold
from image_preprocessing import strip_bmp_header
from camera import Camera, PixelFormat, FrameSize
import cnn_inference
import network
import socket
import time
import gc

# WiFi credentials
SSID = "RedRover"
PASSWORD = ""
LAPTOP_IP = "10.49.100.6"  # Update this!
PORT = 5001

# Connect WiFi
wlan = network.WLAN(network.STA_IF)
wlan.active(False)
time.sleep(0.5)
wlan.active(True)
time.sleep(0.5)
wlan.connect(SSID, PASSWORD)
while not wlan.isconnected():
    time.sleep(0.5)
print("WiFi connected! IP:", wlan.ifconfig()[0])

# Camera setup
CAMERA_PARAMETERS = {
    "data_pins": [15, 17, 18, 16, 14, 12, 11, 48],
    "vsync_pin": 38,
    "href_pin": 47,
    "sda_pin": 40,
    "scl_pin": 39,
    "pclk_pin": 13,
    "xclk_pin": 10,
    "xclk_freq": 20000000,
    "powerdown_pin": -1,
    "reset_pin": -1,
    "frame_size": FrameSize.R96X96,
    "pixel_format": PixelFormat.GRAYSCALE
}

cam = Camera(**CAMERA_PARAMETERS)
cam.init()
cam.set_bmp_out(True)
print("Camera ready!")

# Connect to laptop
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((LAPTOP_IP, PORT))
print("Connected to laptop!")

while True:
    try:
        buf = cam.capture()
        gc.collect()

        resized = resize_96x96_to_32x32_and_threshold(bytearray(buf), 128)
        gc.collect()

        pixels = strip_bmp_header(resized)
        gc.collect()

        img_2d = [[pixels[y*32 + x] for x in range(32)] for y in range(32)]

        label, probs = cnn_inference.predict(img_2d)
        gc.collect()

        # Send label length, label, then image size and image
        label_bytes = label.encode()
        client.send(len(label_bytes).to_bytes(1, 'big'))
        client.send(label_bytes)
        img_size = len(resized)
        client.send(img_size.to_bytes(4, 'big'))
        client.send(bytes(resized))
        print("Sent:", label)

    except Exception as e:
        print("Error:", e)
        break