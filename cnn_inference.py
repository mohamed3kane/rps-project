import json
import math

_weights = None

def load_weights():
    global _weights
    if _weights is None:
        print("Loading weights...")
        with open('weights.json', 'r') as f:
            _weights = json.load(f)
        print("Weights loaded!")
    return _weights

def relu(x):
    return max(0, x)

def softmax(x):
    max_x = max(x)
    exp_x = [math.exp(i - max_x) for i in x]
    sum_exp = sum(exp_x)
    return [i / sum_exp for i in exp_x]

def conv2d(input_data, weights, biases, padding=1):
    in_h = len(input_data)
    in_w = len(input_data[0])
    in_c = len(input_data[0][0])
    kh = len(weights)
    kw = len(weights[0])
    out_c = len(weights[0][0][0])
    out_h, out_w = in_h, in_w

    output = [[[0.0] * out_c for _ in range(out_w)] for _ in range(out_h)]

    for oc in range(out_c):
        for oh in range(out_h):
            for ow in range(out_w):
                val = biases[oc]
                for ic in range(in_c):
                    for khi in range(kh):
                        for kwi in range(kw):
                            ih = oh + khi - padding
                            iw = ow + kwi - padding
                            if 0 <= ih < in_h and 0 <= iw < in_w:
                                val += input_data[ih][iw][ic] * weights[khi][kwi][ic][oc]
                output[oh][ow][oc] = relu(val)
    return output

def maxpool2d(input_data, pool_size=2):
    in_h = len(input_data)
    in_w = len(input_data[0])
    in_c = len(input_data[0][0])
    out_h, out_w = in_h // pool_size, in_w // pool_size
    output = [[[0.0] * in_c for _ in range(out_w)] for _ in range(out_h)]
    for c in range(in_c):
        for oh in range(out_h):
            for ow in range(out_w):
                max_val = -999999
                for ph in range(pool_size):
                    for pw in range(pool_size):
                        v = input_data[oh*pool_size+ph][ow*pool_size+pw][c]
                        if v > max_val:
                            max_val = v
                output[oh][ow][c] = max_val
    return output

def flatten(input_data):
    result = []
    for row in input_data:
        for col in row:
            for val in col:
                result.append(val)
    return result

def dense(input_data, weights, biases, activation='relu'):
    output = []
    for i in range(len(biases)):
        val = biases[i]
        for j in range(len(input_data)):
            val += input_data[j] * weights[j][i]
        if activation == 'relu':
            val = relu(val)
        output.append(val)
    return output

def predict(image_array):
    w = load_weights()

    x = [[[pixel/255.0] for pixel in row] for row in image_array]

    print("Conv1...")
    x = conv2d(x, w['layer_0'][0], w['layer_0'][1])
    x = maxpool2d(x)
    print("Conv2...")
    x = conv2d(x, w['layer_2'][0], w['layer_2'][1])
    x = maxpool2d(x)
    print("Conv3...")
    x = conv2d(x, w['layer_4'][0], w['layer_4'][1])
    x = maxpool2d(x)
    print("Dense...")
    x = flatten(x)
    x = dense(x, w['layer_7'][0], w['layer_7'][1])
    x = dense(x, w['layer_9'][0], w['layer_9'][1], activation='softmax')
    x = softmax(x)

    classes = ['rock', 'paper', 'scissors']
    max_idx = x.index(max(x))
    return classes[max_idx], x