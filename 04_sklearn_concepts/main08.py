#퍼셉트론(뉴런), MLP

import numpy as np


def step(x):
    if x > 0: return 1
    else: return 0

##and/or 연산
# # x = [1,1]
# # w = [0,0]
# x = np.array([1,0])     #입력값 x1,x2
# w = np.array([1,1])     #가중치 w1,w2
# b = -0.2                   #편향
#
# z = np.dot(x,w) + b
# y = step(z)
#
# print(y)

def perceptron(x,w,b):
    z = np.dot(x,w) + b
    y = step(z)
    return y

def AND(x1,x2):
    y = perceptron(np.array([x1,x2]), np.array([0.5, 0.5]), -0.9)
    return y

def OR(x1,x2):
    y = perceptron(np.array([x1,x2]), np.array([0.5, 0.5]), -0.4)
    return y

def NAND(x1,x2):
    y = not AND(x1,x2)
    return int(y)

##MLP  - t1, t2를 만들어서 층을 쌓았음
def XOR(x1,x2):
    t1 = NAND(x1,x2)
    t2 = OR(x1,x2)
    y = AND(t1,t2)
    return y


# print("=== and ===")
# print(AND(0,0))
# print(AND(0,1))
# print(AND(1,0))
# print(AND(1,1))
#
# print("=== or ===")
# print(OR(0,0))
# print(OR(0,1))
# print(OR(1,0))
# print(OR(1,1))
#
# print("=== NAND ===")
# print(NAND(0,0))
# print(NAND(0,1))
# print(NAND(1,0))
# print(NAND(1,1))
#
# print("=== XOR ===")
# print(XOR(0,0))
# print(XOR(0,1))
# print(XOR(1,0))
# print(XOR(1,1))

def print_gate(name, func):
    print(f"=== {name} ===")
    print(func(0,0), func(0,1), func(1,0), func(1,1), sep="\n")
    print()

print_gate("and", AND)
print_gate("or", OR)
print_gate("NAND", NAND)
print_gate("XOR", XOR)


