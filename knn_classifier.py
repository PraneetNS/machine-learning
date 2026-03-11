import numpy as np
from collections import Counter

X_train = np.array([
    [9,1],
    [8,2],
    [1,9],
    [2,8],
    [7,3]
])

y_train = np.array([0,0,1,1,0])

k = 3

def distance(a,b):
    return np.sqrt(np.sum((a-b)**2))

def predict(x):

    distances = []

    for i in range(len(X_train)):
        d = distance(x, X_train[i])
        distances.append((d, y_train[i]))

    distances.sort()

    neighbors = distances[:k]
    labels = [n[1] for n in neighbors]

    return Counter(labels).most_common(1)[0][0]

user = np.array([6,3])

print("Predicted user type:", predict(user))