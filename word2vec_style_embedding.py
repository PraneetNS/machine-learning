import numpy as np

vocab = ["I","like","AI","love","ML"]

word_to_idx = {w:i for i,w in enumerate(vocab)}

pairs = [
("I","like"),
("I","love"),
("like","AI"),
("love","ML")
]

V=len(vocab)
embed_dim=3

W1=np.random.randn(V,embed_dim)
W2=np.random.randn(embed_dim,V)

lr=0.01

for epoch in range(1000):

    loss=0

    for w,c in pairs:

        x=np.zeros(V)
        x[word_to_idx[w]]=1

        h=x.dot(W1)
        u=h.dot(W2)

        y_pred=np.exp(u)/np.sum(np.exp(u))

        y=np.zeros(V)
        y[word_to_idx[c]]=1

        loss-=np.log(y_pred[word_to_idx[c]])

        grad=y_pred-y

        W2-=lr*np.outer(h,grad)
        W1-=lr*np.outer(x,grad.dot(W2.T))

print("Embedding for 'AI':")
print(W1[word_to_idx["AI"]])