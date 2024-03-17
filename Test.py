from qutip import *
import numpy as np
import matplotlib.pyplot as plt

print(Qobj())
print(Qobj([[1],[2],[3],[4],[5]]))
x = np.array([[1, 2, 3, 4, 5]])
print(Qobj(x))
r = np.random.rand(4, 4)
print(Qobj(r))

S_x = jmat(1,'x')
print(S_x)
S_y = jmat(1,'y')
print(commutator(S_x,S_y,'anti'))

print(S_x * S_x - S_y * S_y)
print(S_x ** 2 - S_y ** 2)

print(S_y.dag())
eigval,eigket = S_y.eigenstates()
print(eigket)
print(eigval)
print(S_y.tr())

         