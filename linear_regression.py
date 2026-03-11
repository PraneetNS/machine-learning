import numpy as np

# size of house
X = np.array([500, 800, 1000, 1200, 1500], dtype=float)

# price
y = np.array([150000, 220000, 300000, 340000, 400000], dtype=float)

w = 0.0
b = 0.0
lr = 0.0000001
epochs = 2000

n = len(X)

for epoch in range(epochs):

    # prediction
    y_pred = w * X + b

    # loss
    loss = np.mean((y_pred - y) ** 2)

    # gradients
    dw = (2/n) * np.sum((y_pred - y) * X)
    db = (2/n) * np.sum(y_pred - y)

    # update
    w -= lr * dw
    b -= lr * db

print("Weight:", w)
print("Bias:", b)

test_size = 1100
prediction = w * test_size + b

print("Predicted price:", prediction)