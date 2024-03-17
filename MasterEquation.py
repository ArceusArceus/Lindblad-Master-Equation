import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# 系统参数
levels = 3  # 能级数
gamma = np.array([[0,98,130],[98,0,100],[130,100,0]])  # 退极化率

# 定义哈密顿量
D = 2 * np.pi * 2.87 * (10**9)
Field = 510 * 2.8 * (10**6)
H = D * (jmat(1,'z') * jmat(1,'z') - 1/3) + Field * jmat(1,'z')

# 定义初始状态（假设实验操控+1和0能级）
psi0 = Qobj([[1],[1],[0]]) / np.sqrt(2)


# 定义时间点
tlist = np.linspace(0, 10 ** (-4), 100000) 

# 没有退极化的态演化
result0 = mesolve(H, psi0, tlist, [],[])


# 定义Lindblad算符
L = []
for j in range(levels):
    for k in range(j + 1, levels):
        L.append(np.sqrt(gamma[j,k]) * basis(levels, j) * basis(levels, k).dag())  # j->k跃迁
        L.append(np.sqrt(gamma[j,k]) * basis(levels, k) * basis(levels, j).dag())  # k->j跃迁

# 求解Lindblad主方程
result = mesolve(H, psi0, tlist, L, [])

# 绘制各能级占据概率随时间的变化

Proj = []
for n in range(len(tlist)):
    state0 = result0.states[n]
    # Proj_Oper = state0 * state0.dag()
    state = result.states[n]
    Proj.append(fidelity(state0,state))
               


plt.figure(figsize=(8, 5))
plt.plot(tlist, Proj, label=f"Level")
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()
