import serial
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from utils.torch_utils import BaseModule
import sys
import textwrap
# Define model
class NetDQN(BaseModule):
    def __init__(self, n_in, n_out, intra_layers, activation_type="TANH", reset_type="XAVIER", normalize=None):
        super(NetDQN, self).__init__(activation_type, reset_type, normalize)
        all_layers = [n_in] + intra_layers + [n_out]
        self.layers = []
        for i in range(0, len(all_layers) - 2):
            module = torch.nn.Linear(all_layers[i], all_layers[i + 1])
            self.layers.append(module)
            self.add_module("h_" + str(i), module)
        self.predict = torch.nn.Linear(all_layers[-2], all_layers[-1])

    def forward(self, x):
        print("----------- foward")
        if self.normalize:
            x = (x.float() - self.mean.float()) / self.std.float()
        for layer in self.layers:
            print("---------------")
            print("x={}".format(x))
            print("layer.weight= {}".format(layer.weight))
            x = layer(x)
            print("layer(x)={}".format(x))
            x = self.activation(x)
            print("activation(x) = {}".format(x))
        print("x before predict = {}".format(x))
        print("predict weight ={}".format(self.predict.weight))
        x = self.predict(x)
        print("x after predict = {}".format(x))
        print("------------ end forward")
        return x.view(x.size(0), -1)

    def toJson(self):
        print(self)
        weights=[]
        for layer in self.layers:
            print(layer.weight)
            weights.append(layer.weight.tolist())
        print("predict : ")
        print(self.predict.weight)
        weights.append(self.predict.weight.tolist())
        bias=[]
        for layer in self.layers:
            bias.append(layer.bias.tolist())
        bias.append(self.predict.bias.tolist())
        nn_json={
            "weights": weights,
            "bias":bias
        }
        return nn_json
model = NetDQN(10,4,[7,5])

# model = NetDQN(2,1,[2,2])


model.reset()
print("y={}".format(model.forward(torch.tensor([float(x) for x in range(10)]))))


arduino_serial = serial.Serial('/dev/ttyACM1',115200)
time.sleep(1)
#x = json.dumps({'bias':[i for i in range(0,10)]})
x = json.dumps(model.toJson())
lenx = len(x)
lenlenx = len(str(lenx))

for i in range(1):
    message = "<"+str(lenlenx) + str(lenx)+x+">"
    # print("Message total size={}".format(len(message)))
    print(message)
    arduino_serial.write(message)
    #arduino_serial.write("<15hello>")

    # chunks = textwrap.wrap(message, 100)
    # for chunk in chunks:
    #     print("sending chunk (size={}) : {}".format(len(chunk),chunk))
    #     arduino_serial.write(chunk)
