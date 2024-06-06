import numpy as np
from util import *
from keygen import keygen
from enc import encrypt, buildGadget
from dec import decrypt

keys=keygen(24)

for a,b in [(1,1),(17,19),(34,62),(21,20)]:
    ca=encrypt(keys,a)
    cb=encrypt(keys,b)
    a_b=a+b
    ca_cb=(ca+cb)%keys.q
    d_ca_cb=decrypt(keys,ca_cb)

    print(" "*12 + "Expected %d" % a_b)
    print(" "*12 + "Received %d" % d_ca_cb)

    if a_b==d_ca_cb:
        print(" "*12 + "\x1B[32;1mPassed\x1B[0m")
    else:
        print(" "*12 + "\x1B[31;1mFailed\x1B[0m")

#a*b
G=buildGadget(keys.l, keys.n)
ca=encrypt(keys,a)
cb=encrypt(keys,b)

a_b=a*b
binary_digit=keys.l
rows, columns=cb.shape
G_cb=np.zeros((rows*keys.l,columns))

for i in range(rows):
    for j in range(columns):
        binary_represent=format(cb[i][j],f'0{binary_digit}b')
        for k in range(len(binary_represent)-1,-1,-1):
            G_cb[i*binary_digit+len(binary_represent)-1-k][j]=int(binary_represent[k])


ca_cb=(np.dot(ca,G_cb))%keys.q
d_ca_cb=decrypt(keys,ca_cb)

print(" "*12 + "Expected %d" % a_b)
print(" "*12 + "Received %d" % d_ca_cb)

if a_b==d_ca_cb:
    print(" "*12 + "\x1B[32;1mPassed\x1B[0m")
else:
    print(" "*12 + "\x1B[31;1mFailed\x1B[0m")
