import socket
import sys
import time

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
        print(recvall(s).decode('utf-8').strip())
        cmd = input()
        s.send(cmd.encode('utf-8'))

        # train model
        if cmd == '1':
            while True:
                msg = recvall(s).decode('utf-8').strip()
                if len(msg) == 0:
                    break
                print(msg)
                inp = input()
                inp = inp if inp else 'default_opt'
                s.send(inp.encode('utf-8'))

            # TODO: decide to use which model and send encryrpted dataset

            success = s.recv(1024).decode('utf-8')
            if success == '1': # successful
                print('Training successful')
                # save model parameters to file
                time.sleep(0.5)
                data = recvall(s)
                
                print('Please input the desired path to save the parameters: ')
                path = input()
                with open(path, 'wb') as f:
                    f.write(data)
            else:
                print('Training failed')
        # load model
        elif cmd == '2':
            success = s.recv(1024).decode('utf-8')
            # TODO: load model parameters from file

            if success == '1': # successful
                # load model parameters from file
                print('Loading successful')
            else:
                print('Loading failed')
        # predict
        elif cmd == '3':
            success = s.recv(1024).decode('utf-8')
            # TODO: take input and send it to the server to make prediction

            if success == '1': # successful
                print('Predicting successful')
                # predict
            else:
                print('Loading failed')
        elif cmd == '4':
            print(s.recv(1024).decode('utf-8'))
            exit(0)
        else:
            print('Invalid command')
        time.sleep(0.5) # prevent client read 2 messages at the same time
