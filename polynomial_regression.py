import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# house size
X = np.array([[500],[800],[1000],[1200],[1500]])

# price
y = np.array([150000,220000,300000,340000,400000])

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

w = np.zeros(X_poly.shape[1])
lr = 1e-12
epochs = 5000

n = len(y)

for epoch in range(epochs):

    y_pred = X_poly.dot(w)

    loss = np.mean((y_pred-y)**2)

    grad = (2/n)*X_poly.T.dot(y_pred-y)

    w -= lr*grad

print("Weights:",w)

test = np.array([[1100]])
test_poly = poly.transform(test)

prediction = test_poly.dot(w)

print("Predicted price:",prediction)