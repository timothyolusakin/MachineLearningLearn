import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


X,Y = np.loadtxt("pizza.txt",skiprows=1, unpack=True)

def predict(X : int,w:int,b:int):
    return X * w + b

def loss(X:int,Y:int,w:int,b:int):
    return np.average((predict(X,w,b) - Y) ** 2)

def train(X:int,Y:int,iterations:int,lr:int):
    w=b=0
    for i in range(iterations):
        current_loss = loss(X,Y,w,b)
        print("Iteration %4d => Loss: %.6f" %(i,current_loss))
        if loss(X,Y,w+lr,b) < current_loss:
            w += lr
        elif loss(X,Y,w-lr,b)  < current_loss:
            w -= lr
        elif loss(X,Y,w,b+lr) < current_loss:
            b += lr
        elif loss(X,Y,w,b-lr)  < current_loss:
            b -= lr
        else:
            return w,b
    raise Exception("Couldn't converge within %d iterations"% iterations)

w,b = train(X,Y,iterations=10000,lr=0.01)
print("\nw=%.3f, b=%.3f"% (w,b))
print("Prediction: x=%d => y=%.2f"% (20, predict(20,w,b)))

### With gradient descent #####
def gradient(X:int, Y:int,w:int,b:int):
    w_gradient = 2 * np.average(X*(predict(X,w,b)-Y))
    b_gradient = 2 * np.average(predict(X,w,b)-Y)
    return (w_gradient,b_gradient)

def train_with_gradient(X:int,Y:int,iterations:int, lr:int):
    w=b=0
    for i in range(iterations):
        print("Iteration %4d => Loss:%.10f"%(i,loss(X,Y,w,b)))
        w_gradient, b_gradient = gradient(X,Y,w,b)
        w -= w_gradient*lr
        b -= b_gradient*lr
    return w,b
#w,b = train_with_gradient(X,Y,iterations=20000,lr=0.001)
#print("\nw=%.10f, b=%.10f"%(w,b))
#print("Prediction: x=%d => y=%.2f"% (20, predict(20,w,b)))