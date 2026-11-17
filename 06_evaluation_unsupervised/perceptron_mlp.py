import numpy as np


#퍼셉트론, mlp

def step(x):
    if x>0: return 1
    else: return 0

def perceptron(x,w,b):
    z = np.dot(x,w)+b
    y = step(z)
    return y

def AND(x1,x2):
    y = perceptron(np.array([x1,x2]), np.array([0.5,0.5]), -0.9)
    return y

def OR(x1,x2):
    y = perceptron(np.array([x1,x2]), np.array([0.5,0.5]), -0.4)
    return y

def NAND(x1,x2):
    y = not AND(x1,x2)
    return int(y)

##MLP
def XOR(x1,x2):
    t1 = NAND(x1,x2)
    t2 = OR(x1,x2)
    y = AND(t1,t2)
    return y

def logic_gate(name, func):
    print(f"=== {name} ===")
    print(func(0,0), func(0,1), func(1,0), func(1,1),
          sep="\n")

logic_gate("AND", AND)
logic_gate("OR", OR)
logic_gate("NAND", NAND)
logic_gate("XOR", XOR)


