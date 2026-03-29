import numpy as np

np.random.seed(0)

X = np.array([
[1,0,1,0],
[0,1,0,1],
[1,1,0,0],
[0,0,1,1]
])

W1 = np.random.randn(4,2)
W2 = np.random.randn(2,4)

lr = 0.1

for epoch in range(5000):

    hidden = 1/(1+np.exp(-X.dot(W1)))
    output = 1/(1+np.exp(-hidden.dot(W2)))

    loss = np.mean((X-output)**2)

    d_output = (output-X)*output*(1-output)
    dW2 = hidden.T.dot(d_output)

    d_hidden = d_output.dot(W2.T)*hidden*(1-hidden)
    dW1 = X.T.dot(d_hidden)

    W1 -= lr*dW1
    W2 -= lr*dW2

print("Reconstructed:")
print(output)