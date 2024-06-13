import socket
import sys
import json
from time import time
import time as t
import ast

import torch
import tenseal as ts
import pandas as pd
import random
import pickle
from sklearn.model_selection import train_test_split



class LR(torch.nn.Module):

    def __init__(self, n_features):
        super(LR, self).__init__()
        self.lr = torch.nn.Linear(n_features, 1)

    def forward(self, x):
        out = torch.sigmoid(self.lr(x))
        return out

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
        #if self._count == 0:
            #raise RuntimeError("You should at least run one forward iteration")
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
        print(out)
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



def save_dataset(s, filename):
    with open(filename, 'wb') as f:
        f.write(s.recv(1024))
        s.settimeout(0.1)
        while True:
            try:
                f.write(s.recv(1024))
            except:
                s.settimeout(None)
                return

def get_model_args(conn):
    model_id = 0
    args = []
    with open('models.json', 'r') as f:
        models = json.load(f)
        prompt = 'Choose a model from below: \n'
        while True:
            for i, model in enumerate(models['models']):
                prompt += f"{i+1}: {model['name']}\n"
            conn.send(prompt.encode('utf-8'))
            model_id_str = conn.recv(1024).decode('utf-8').strip()
            if model_id_str.isdigit() and 0 < int(model_id_str) <= len(models['models']):
                model_id = int(model_id_str) - 1
                break
            else:
                conn.send(b'Invalid model ID\n')
        for i, param in enumerate(models['models'][model_id]['params']):
            conn.send(f"Enter value for parameter {param}: [{models['models'][model_id]['params'][param]}]".encode('utf-8'))
            param_value = conn.recv(1024).decode('utf-8').strip()
            if param_value != 'default_opt':
                param_value = ast.literal_eval(param_value)
            else:
                param_value = models['models'][model_id]['params'][param]

            args.append(param_value)
            
    return args

def handle_train(conn):
    # after training, save the model parameters to a file
    file_path = 'test.txt'
    success = True
    
    # query for model type and hyper-parameters
    model_args = get_model_args(conn)

    t.sleep(0.5) # let client know that the server is ready to receive the dataset

    #try:
    conn.send(b'please input the path to the dataset (x_train): ')
    save_dataset(conn, 'x_train.csv')
    conn.send(b'please input the path to the dataset (y_train): ')
    save_dataset(conn, 'y_train.csv')

    path_x='x_train.csv'
    path_y='y_train.csv'

    ctx_training = ts.context(ts.SCHEME_TYPE.CKKS, int(model_args[1]), -1, model_args[2])
    ctx_training.global_scale = 2 ** 21
    ctx_training.generate_galois_keys()

    with open(path_x, 'rb') as f:
        enc_x_=[ts.ckks_vector_from(ctx_training,_)for _ in f.read().split(b'\n\n\n\n\n\n')][:-1]


    with open(path_y, 'rb') as f:
        enc_y_=[ts.ckks_vector_from(ctx_training,_)for _ in f.read().split(b'\n\n\n\n\n\n')][:-1]


    eelr = EncryptedLR(LR(model_args[3]))
    for epoch in range(model_args[0]):
        eelr.encrypt(ctx_training)
        
        for enc_x, enc_y in zip(enc_x_, enc_y_):
            enc_out = eelr.forward(enc_x)
            eelr.backward(enc_x, enc_out, enc_y)
        eelr.update_parameters()
        eelr.decrypt()
    with open('eelr.pkl','wb') as file:
        pickle.dump(eelr,file)


    file_path='eelr.pkl'

    # dataset is now saved in dataset.txt
    # TODO: code for training here
    # model_args saves the hyper-parameters for the model
    # save the trained model parameters to a file
    print('Model trained')
    '''
    except:
        success = False
    '''
    return file_path, success

def handle_load(conn):
    success = True

    try:
        # TODO: code for loading here
        print('Model loaded')
    except:
        success = False

    return success

def handle_predict(conn):
    result = ''
    success = True
    
    try:
        # TODO: code for predicting here
        print('Prediction made')
    except:
        success = False

    return result, success

if __name__ == '__main__':
    if len(sys.argv) > 2:
        print('Usage: python server.py [port]')
        sys.exit(1)

    host = '0.0.0.0'
    port = 12345 if len(sys.argv)==1 else int(sys.argv[1])

    # socket configuration - server
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    s.listen(1)

    conn, addr = s.accept()

    while True:
        try:
            op = conn.recv(1024)
            if op == b'1':
                file_path, success = handle_train(conn)
                t.sleep(0.5) # sleep so that client to receive the success message
                if success:
                    conn.send(b'1')
                    t.sleep(0.5)
                    with open(file_path, 'rb') as f:
                        conn.sendfile(f)
                else:
                    conn.send(b'0')
                t.sleep(0.5) # prevent client read 2 messages at the same time
            else:
                conn.send(b'Goodbye!')
                exit(0)
        except BrokenPipeError:
            print('Connection closed by client')
            break