# NN_MNIST
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import sys
sys.path.append('/kaggle/input/digit-recognizer')

input_dir = '/kaggle/input'
for root, dirs, files in os.walk(input_dir):
    for dir_name in dirs:
        print(f"Directory: {dir_name}")
    for file_name in files:
        print(f"File: {file_name}")

        
train_df=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
print(train_df.head())

test_df=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
print(test_df.head())

X_train = train_df.drop(columns=['label']).values
y_train = train_df['label'].values
X_test = test_df.values

# Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encode the labels
def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]

y_train_encoded = one_hot_encode(y_train, 10)
X_train, X_val, y_train_encoded, y_val_encoded = train_test_split(X_train, y_train_encoded, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}")  # Should be (num_samples, num_features)
print(f"y_train_encoded shape: {y_train_encoded.shape}")  # Should be (num_samples, num_classes)
num_features = X_train.shape[1]
num_classes = y_train_encoded.shape[1]

print(f"Number of features: {num_features}")
print(f"Number of classes: {num_classes}")


nn_architecture = [
    {"input_dim": X_train.shape[1], "output_dim": 128, "activation": "relu"},
    {"inpudef init_layers(nn_architecture, seed=99):
    np.random.seed(seed)
    params_values = {}
    ]

def init_layers(nn_architecture, seed=99):
    np.random.seed(seed)
    params_values = {}
    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        input_dim = layer["input_dim"]
        output_dim = layer["output_dim"]
        params_values['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
        params_values['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1
    return params_values


params_values = init_layers(nn_architecture, seed=2)
for key, value in params_values.items():
    print(f"{key} shape: {value.shape}")


def sigmoid(z):
    return 1/(1+np.exp(-z))
def relu(z):
    return np.maximum(0,z)
def sigmoid_backwards(dA,z):
    sig=sigmoid(z)
    return dA*sig*(1-sig)
def relu_backwards(dA, z):
    dZ = np.array(dA, copy=True)
    dZ[z <= 0] = 0
    return dZ
def linear(z):
    return z
def linear_backwards(dA,z):
    dZ=dA
    return dA
def tanh(z):
    return (2/(1+np.exp(-2*z)))-1
def tanh_backwards(dA,z):
    y=tanh(z)
    return 1-(y)^2
def softmax(z):
    z_shifted=z-np.max(z,axis=0, keepdims=True)
    exp_z=np.exp(z_shifted)
    A=exp_z/(np.sum(exp_z,axis=0,keepdims=True))
    return A
def softmax_backwards(dA, z):
    A = softmax(z)
    dZ = A - dA
    return dZ


def single_layer_forward_prop(A_prev, W_curr, B_curr, activation):
    print(f"A_prev shape: {A_prev.shape}")
    print(f"W_curr shape: {W_curr.shape}")
    print(f"B_curr shape: {B_curr.shape}")
    Z_curr = np.dot( A_prev,W_curr.T) + B_curr.T
    if activation == "relu":
        activation_func = relu
    elif activation == "linear":
        activation_func = lambda x: x
    elif activation == "tanh":
        activation_func = tanh
    elif activation == "sigmoid":
        activation_func = sigmoid
    elif activation == "softmax":
        activation_func = softmax
    else:
        raise Exception("Non-supported activation function")
    return activation_func(Z_curr), Z_curr


def full_forward_propagation(X, params_values, nn_architecture):
    memory = {}
    A_curr = X
    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        A_prev = A_curr
        activ_func_curr = layer["activation"]
        W_curr = params_values["W" + str(layer_idx)]
        B_curr = params_values["b" + str(layer_idx)]
        A_curr, Z_curr = single_layer_forward_prop(A_prev, W_curr, B_curr, activ_func_curr)
        memory["A" + str(layer_idx)] = A_curr
        memory["Z" + str(layer_idx)] = Z_curr
    return A_curr, memory


  def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation):
    m = A_prev.shape[1]
    if activation == "relu":
        backward_activation_function = relu_backwards
    elif activation == "linear":
        backward_activation_function = lambda dA, Z: dA
    elif activation == "tanh":
        backward_activation_function = tanh_backwards
    elif activation == "sigmoid":
        backward_activation_function = sigmoid_backwards
    elif activation == "softmax":
        backward_activation_function = softmax_backwards
    else:
        raise Exception("Non-supported activation function")
    dZ_curr = backward_activation_function(dA_curr, Z_curr)
    dW_curr = np.dot(dZ_curr, A_prev) / A_prev.shape[1]
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / A_prev.shape[1]
    dA_prev = np.dot(W_curr.T, dZ_curr)
    return dA_prev, dW_curr, db_curr


def full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture):
    grads_values = {}
    m = Y.shape[1]
    dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))
    for idx, layer in reversed(list(enumerate(nn_architecture))):
        layer_idx = idx + 1
        A_prev = memory["A" + str(layer_idx - 1)]
        Z_curr = memory["Z" + str(layer_idx)]
        W_curr = params_values["W" + str(layer_idx)]
        b_curr = params_values["b" + str(layer_idx)]
        dA_prev, dW_curr, db_curr = single_layer_backward_propagation(
            dA_prev, W_curr, b_curr, Z_curr, A_prev, layer["activation"]
        )
        print(f"Layer {layer_idx}: A_prev shape = {A_prev.shape}, W_curr shape = {W_curr.shape}")
        grads_values["dW" + str(layer_idx)] = dW_curr
        grads_values["db" + str(layer_idx)] = db_curr
    return grads_values

    
def full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture):
    grads_values = {}
    m = Y.shape[1]
    dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))
    for idx, layer in reversed(list(enumerate(nn_architecture))):
        layer_idx = idx + 1
        A_prev = memory["A" + str(layer_idx - 1)]
        Z_curr = memory["Z" + str(layer_idx)]
        W_curr = params_values["W" + str(layer_idx)]
        b_curr = params_values["b" + str(layer_idx)]
        dA_prev, dW_curr, db_curr = single_layer_backward_propagation(
            dA_prev, W_curr, b_curr, Z_curr, A_prev, layer["activation"]
        )
        print(f"Layer {layer_idx}: A_prev shape = {A_prev.shape}, W_curr shape = {W_curr.shape}")
        grads_values["dW" + str(layer_idx)] = dW_curr
        grads_values["db" + str(layer_idx)] = db_curr
    return grads_values
    
def update(params_values, grads_values, nn_architecture, learning_rate):
    for layer_idx, layer in enumerate(nn_architecture):
        layer_idx += 1
        params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]
        params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)] 
    return params_values
def convert_prob_into_class(probabilities):
    return np.argmax(probabilities, axis=1)
def convert_one_hot_to_class(Y_one_hot):
    return np.argmax(Y_one_hot, axis=1)
def get_cost_function(Y_hat,Y):
    m=Y.shape[0]
    cost = -np.mean(np.sum(Y * np.log(Y_hat + 1e-8), axis=1))
    return np.squeeze(cost) 
def get_accuracy_value(Y_hat, Y):
    Y_hat = convert_prob_into_class(Y_hat)
    Y=convert_one_hot_to_class(Y)
    return (Y_hat == Y).all(axis=0).mean()

def train(X, Y, nn_architecture, epochs, learning_rate):
    params_values = init_layers(nn_architecture, seed=2)
    cost_history = []
    accuracy_history = []
    for i in range(epochs):
        Y_hat, memory = full_forward_propagation(X, params_values, nn_architecture)
        cost = get_cost_function(Y_hat, Y)
        cost_history.append(cost)
        accuracy = get_accuracy_value(Y_hat, Y)
        accuracy_history.append(accuracy)
        grads_values = full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture)
        params_values = update(params_values, grads_values, nn_architecture, learning_rate)
        print(f"Epoch {i + 1}/{epochs}, Cost: {cost}, Accuracy: {accuracy}")
    return params_values, cost_history, accuracy_history

epochs = 20
learning_rate = 0.01
params_values, cost_history, accuracy_history = train(X_train, y_train_encoded, nn_architecture, epochs, learning_rate)

# Evaluate the model
Y_hat, _ = full_forward_propagation(X_test, params_values, nn_architecture)
test_accuracy = get_accuracy_value(Y_hat, np.argmax(y_test, axis=1))
print(f'Test Accuracy: {test_accuracy}')
