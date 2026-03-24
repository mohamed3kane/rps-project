from image_preprocessing import resize_96x96_to_32x32_and_threshold
from image_preprocessing import strip_bmp_header
from camera import Camera, PixelFormat, FrameSize
import cnn_inference
import gc

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

print("Capturing image...")
buf = cam.capture()
gc.collect()

resized = resize_96x96_to_32x32_and_threshold(bytearray(buf), 128)
gc.collect()

pixels = strip_bmp_header(resized)
gc.collect()

img_2d = [[pixels[y*32 + x] for x in range(32)] for y in range(32)]

print("Running inference...")
label, probs = cnn_inference.predict(img_2d)
print("Prediction:", label)
print("rock={:.2f} paper={:.2f} scissors={:.2f}".format(probs[0], probs[1], probs[2]))