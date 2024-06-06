from util import *
import numpy as np
from scipy.linalg import block_diag

def buildGadget(l,n):
    g=2**np.arange(l)
    return block_diag(*[g for null in range(n)])

def encrypt(keys,message):
    stat("Encrypting  message")
    R=np.random.randint(2,size=(keys.m,keys.m),dtype=np.int64).astype(keys.datatype)
    G=buildGadget(keys.l, keys.n)
    return (np.dot(keys.PK,R)+message*G)%keys.q
