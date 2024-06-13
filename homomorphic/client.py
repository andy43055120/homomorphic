import socket
import sys
import time
import pickle
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import tenseal as ts
import random

class EncryptedLR:
    
    def __init__(self, torch_lr):
        self.weight = torch_lr.lr.weight.data.tolist()[0]
        self.bias = torch_lr.lr.bias.data.tolist()
        # we accumulate gradients and counts the number of iterations
        self._delta_w = 0
        self._delta_b = 0
        self._count = 0
        
    def forward(self, enc_x):
        #print("test: ", enc_x.shape)
        enc_out = enc_x.dot(self.weight) + self.bias
        enc_out = EncryptedLR.sigmoid(enc_out)
        return enc_out
    
    def backward(self, enc_x, enc_out, enc_y):
        out_minus_y = (enc_out - enc_y)
        # print(enc_x.shape)
        #　print(out_minus_y.shape)
        self._delta_w += enc_x * out_minus_y
        self._delta_b += out_minus_y
        self._count += 1
        
    def update_parameters(self):
        if self._count == 0:
            raise RuntimeError("You should at least run one forward iteration")
        # update weights
        # use a small regularization term to keep the output of the linear layer in the range of the sigmoid approximation
        self.weight -= self._delta_w * (1 / self._count) + self.weight * 0.05
        self.bias -= self._delta_b * (1 / self._count)
        # reset gradient accumulators and iterations count
        self._delta_w = 0
        self._delta_b = 0
        self._count = 0
    
    @staticmethod
    def sigmoid(enc_x):
        # We use the polynomial approximation of degree 3
        # sigmoid(x) = 0.5 + 0.197 * x - 0.004 * x^3
        # from https://eprint.iacr.org/2018/462.pdf
        # which fits the function pretty well in the range [-5,5]
        return enc_x.polyval([0.5, 0.197, 0, -0.004])
    
    def plain_accuracy(self, x_test, y_test):
        # evaluate accuracy of the model on
        # the plain (x_test, y_test) dataset
        w = torch.tensor(self.weight)
        b = torch.tensor(self.bias)
        #print("x_test.matmul(w): ", x_test.float().matmul(w))
        out = torch.sigmoid(x_test.float().matmul(w.float()) + b).reshape(-1, 1)
        correct = torch.abs(y_test - out) < 0.5
        return correct.float().mean()    
    
    def encrypt(self, context):
        self.weight = ts.ckks_vector(context, self.weight)
        self.bias = ts.ckks_vector(context, self.bias)
        
    def decrypt(self):
        self.weight = self.weight.decrypt()
        self.bias = self.bias.decrypt()
        
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    

torch.random.manual_seed(73)
random.seed(73)

cancer = pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/Cancer.csv')

y = []
for i in cancer['diagnosis'].values.tolist():
    if i=='M':
        y.append([1.0])
    else:
        y.append([0.0])

x = cancer.drop(['id','diagnosis','Unnamed: 32', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst'],axis=1)
#print(x.columns)
x = (x - x.mean()) / x.std()
'''
    x = torch.tensor(data.values).float()
    return split_train_test(x, y)
'''

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=252)


x_train = torch.tensor(pd.DataFrame(x_train).values, dtype=float)
x_test = torch.tensor(pd.DataFrame(x_test).values, dtype=float)
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)

def recvall(s):
    ret = b''
    s.settimeout(0.1)
    while True:
        try:
            ret += s.recv(1024)
        except:
            s.settimeout(None)
            return ret

if __name__ == '__main__':
    if len(sys.argv) > 2:
        print('Usage: python client.py [server-port]')
        sys.exit(1)

    host = '127.0.0.1'
    port = 12345 if len(sys.argv)==1 else int(sys.argv[1])

    # socket configuration - server
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))

    while True:
        print('Choose an operation below:\n1. train model\n2. predict\n3. quit')
        cmd = input()
        s.send(cmd.encode('utf-8'))

        # train model
        if cmd == '1':
            # decide to use which model and set parameters
            while True:
                msg = recvall(s).decode('utf-8').strip()
                if len(msg) == 0:
                    break
                print(msg)
                inp = input()
                inp = inp if inp else 'default_opt'
                s.send(inp.encode('utf-8'))

            for i in range(2):
                time.sleep(0.5)
                print(recvall(s).decode('utf-8').strip())
                while True:
                    try:
                        path = input()
                        with open(path, 'rb') as f:
                            # TODO: encrypt dataset here
                            s.sendfile(f)
                            break
                    except:
                        print('Invalid path, please try again')

            success = s.recv(1024).decode('utf-8')
            if success == '1': # successful
                print('Training successful')
                # save model parameters to file
                time.sleep(0.5)
                data = recvall(s)
                

                while True:
                    try:
                        print('Please input the desired path to save the model: ')
                        path = input()
                        with open(path, 'wb') as f:
                            f.write(data)
                        break
                    except:
                        print('Invalid path, please try again')

            else:
                print('Training failed')
        # load model
        # predict
        elif cmd == '2':
            # TODO: unpickling the model and predicting
            with open('trained_model/trained_model','rb') as f:
                model_t=pickle.load(f)
            accuracy = model_t.plain_accuracy(x_test, y_test)
            print("accuracy:")
            print(accuracy)
        # quit
        elif cmd == '3':
            s.send(3)
            print(s.recv(1024).decode('utf-8'))
            exit(0)
        else:
            print('Invalid command')
        time.sleep(0.5) # prevent client read 2 messages at the same time

            
