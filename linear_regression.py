import numpy as np

X = np.array([1,2,3,4,5], dtype=float)
y = np.array([3,5,7,9,11], dtype=float)

w = 0
b = 0
lr = 0.01
epochs = 1000

for epoch in range(epochs):

    # prediction
    y_pred = None

    # loss (MSE)
    loss = None

    # gradients
    dw = None
    db = None

    # update
    w = None
    b = None

print(w, b)