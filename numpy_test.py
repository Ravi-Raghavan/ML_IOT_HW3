import numpy as np

a = np.arange(10)

b  = np.clip(a, 1, 8)

print("A: ", a)
print("B: ", b)

a = np.array([0, 1, 2])

print(a[:, np.newaxis])
print(a[:, np.newaxis].shape)

print(np.tile(a[:, np.newaxis], reps = (2, 5)))


C = np.array([[1, 2, 3], [2, 3, 4], [3,4,5]])
D = np.array([[1, [2], [3]]])
print(C + D)