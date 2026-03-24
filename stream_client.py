import network
import socket
import time
from camera import Camera, GrabMode, PixelFormat, FrameSize, GainCeiling

# WiFi credentials
SSID = "RedRover"
PASSWORD = ""
LAPTOP_IP = "10.49.100.6"
PORT = 5001

# Connect to WiFi
wlan = network.WLAN(network.STA_IF)
wlan.active(False)
time.sleep(0.5)
wlan.active(True)
time.sleep(0.5)
wlan.connect(SSID, PASSWORD)

while not wlan.isconnected():
    time.sleep(0.5)
print("WiFi connected! ESP IP:", wlan.ifconfig()[0])

# Initialize camera
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
try:
    cam.deinit()
except:
    pass
cam.init()
cam.set_bmp_out(True)
print("Camera ready!")

# Connect to laptop server
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((LAPTOP_IP, PORT))
print("Connected to laptop!")

while True:
    buf = cam.capture()
    if buf:
        size = len(buf)
        client.send(size.to_bytes(4, "big"))
        client.send(buf)
        print("Frame sent, size:", size)
    time.sleep(0.1)