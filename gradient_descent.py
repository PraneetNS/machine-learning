x = 8
lr = 0.1
epochs = 50

def f(x):
    return x**2 + 6*x + 9

def grad(x):
    return 2*x + 6

for i in range(epochs):

    g = grad(x)
    x -= lr * g

print("Minimum x:", x)
print("Minimum value:", f(x))