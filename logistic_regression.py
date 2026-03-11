import numpy as np

# feature: number of spam words
X = np.array([1,2,3,4,5,6], dtype=float)

# labels
y = np.array([0,0,0,1,1,1], dtype=float)

w = 0.0
b = 0.0
lr = 0.01
epochs = 3000

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

n = len(X)

for epoch in range(epochs):

    z = w * X + b
    y_pred = sigmoid(z)

    loss = -np.mean(y*np.log(y_pred+1e-9) + (1-y)*np.log(1-y_pred+1e-9))

    dw = (1/n) * np.sum((y_pred - y) * X)
    db = (1/n) * np.sum(y_pred - y)

    w -= lr * dw
    b -= lr * db

print("Weight:", w)
print("Bias:", b)

test_email = 4
prob = sigmoid(w * test_email + b)

print("Spam probability:", prob)