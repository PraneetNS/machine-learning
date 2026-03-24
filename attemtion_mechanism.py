import numpy as np

np.random.seed(0)

# 3 tokens, embedding size 4
X = np.random.randn(3,4)

Wq = np.random.randn(4,4)
Wk = np.random.randn(4,4)
Wv = np.random.randn(4,4)

Q = X.dot(Wq)
K = X.dot(Wk)
V = X.dot(Wv)

scores = Q.dot(K.T) / np.sqrt(4)

# softmax
exp_scores = np.exp(scores - np.max(scores))
weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

output = weights.dot(V)

print("Attention Output:")
print(output)