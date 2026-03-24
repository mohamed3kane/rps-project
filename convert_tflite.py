import keras
import numpy as np
import tensorflow as tf

# Load the model
model = keras.models.load_model('/Users/mohamedkane/Desktop/rps_model_noaug.keras')

# Save as H5 first
model.save('/Users/mohamedkane/Desktop/rps_model.h5')
print("H5 model saved!")

# Convert using concrete function approach
@tf.function(input_signature=[tf.TensorSpec(shape=[1, 32, 32, 1], dtype=tf.float32)])
def predict(x):
    return model(x)

converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [predict.get_concrete_function()]
)
tflite_model = converter.convert()

with open('/Users/mohamedkane/Desktop/rps_model.tflite', 'wb') as f:
    f.write(tflite_model)
print("TFLite model saved!")