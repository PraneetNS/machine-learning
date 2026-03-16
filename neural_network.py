import numpy as np

X = np.array([
[0,0],
[0,1],
[1,0],
[1,1]
])

y = np.array([[0],[1],[1],[0]])

np.random.seed(0)

W1 = np.random.randn(2,4)
b1 = np.zeros((1,4))

W2 = np.random.randn(4,1)
b2 = np.zeros((1,1))

lr = 0.1
epochs = 10000

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_deriv(x):
    return x*(1-x)

for epoch in range(epochs):

    z1 = X.dot(W1) + b1
    a1 = sigmoid(z1)

    z2 = a1.dot(W2) + b2
    y_pred = sigmoid(z2)

    loss = np.mean((y - y_pred)**2)

    dz2 = (y_pred - y) * sigmoid_deriv(y_pred)
    dW2 = a1.T.dot(dz2)
    db2 = np.sum(dz2, axis=0, keepdims=True)

    dz1 = dz2.dot(W2.T) * sigmoid_deriv(a1)
    dW1 = X.T.dot(dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)

    W1 -= lr*dW1
    b1 -= lr*db1
    W2 -= lr*dW2
    b2 -= lr*db2

print("Predictions:")
print(y_pred)