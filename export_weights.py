import os
import numpy as np
from PIL import Image
import keras
import json

model = keras.models.load_model('/Users/mohamedkane/Desktop/rps_model_small.keras')

weights = {}
for i, layer in enumerate(model.layers):
    w = layer.get_weights()
    if w:
        weights[f'layer_{i}'] = [arr.tolist() for arr in w]
        print(f"Layer {i} ({layer.name}): {[arr.shape for arr in w]}")

with open('/Users/mohamedkane/Desktop/weights.json', 'w') as f:
    json.dump(weights, f)

print("Done!")