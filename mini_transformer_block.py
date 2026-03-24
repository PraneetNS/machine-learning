import numpy as np

np.random.seed(0)

X = np.random.randn(3,4)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e, axis=1, keepdims=True)

# Attention
Wq = np.random.randn(4,4)
Wk = np.random.randn(4,4)
Wv = np.random.randn(4,4)

Q = X.dot(Wq)
K = X.dot(Wk)
V = X.dot(Wv)

scores = Q.dot(K.T)/np.sqrt(4)
attn = softmax(scores)
out1 = attn.dot(V)

# Feedforward
W1 = np.random.randn(4,8)
W2 = np.random.randn(8,4)

ff = np.maximum(0, out1.dot(W1))
out2 = ff.dot(W2)

print("Transformer Output:")
print(out2)