import socket
import sys
import json
import time
import ast

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

    time.sleep(0.5) # let client know that the server is ready to receive the dataset

    try:
        conn.send(b'please input the path to the dataset (x_train): ')
        save_dataset(conn, 'x_train.csv')
        conn.send(b'please input the path to the dataset (y_train): ')
        save_dataset(conn, 'y_train.csv')

        # dataset is now saved in dataset.txt
        # TODO: code for training here
        # model_args saves the hyper-parameters for the model
        # save the trained model parameters to a file
        print('Model trained')
    except:
        success = False

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
            conn.send(b'Choose an operation below:\n1. train model\n2. load model\n3. predict\n4. quit')
            r = conn.recv(1024).decode('utf-8').strip()

            if(r == '1'):
                file_path, success = handle_train(conn)
                time.sleep(0.5) # sleep so that client to receive the success message
                if success:
                    conn.send(b'1')
                    time.sleep(0.5)
                    with open(file_path, 'rb') as f:
                        conn.sendfile(f)
                else:
                    conn.send(b'0')
            elif(r == '2'):
                success = handle_load(conn)
                time.sleep(0.5)
                if success:
                    conn.send(b'1')
                else:
                    conn.send(b'0')
            elif(r == '3'):
                success, result = handle_predict(conn)
                time.sleep(0.5)
                if success:
                    conn.send(b'1')
                    conn.send(result.encode('utf-8'))
                else:
                    conn.send(b'0')
            elif(r == '4'):
                conn.send(b'Goodbye!')
                conn.close()
                break
            else:
                conn.send(b'Invalid option\n')
            time.sleep(0.5) # prevent client read 2 messages at the same time
        except BrokenPipeError:
            print('Connection closed by client')
            break