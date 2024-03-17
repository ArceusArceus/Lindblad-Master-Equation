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

Sy10 = Qobj([[0,-1j * 0.5,0],[1j * 0.5,0,0],[0,0,1]])
HalfPi_Sy10 = -1j * 0.5 * np.pi * Sy10
# Sz10 = np.array([[1,0,0],[0,-1,0],[0,0,1]])



# 定义初始状态（假设实验操控+1和0能级）
psi0 = Qobj([[1],[1],[0]]) / np.sqrt(2)


# 定义时间点
tlist = np.linspace(0, 10 ** (-3), 1000000) 


# 定义Lindblad算符
L = []
for j in range(levels):
    for k in range(j + 1, levels):
        L.append(np.sqrt(gamma[j,k]) * basis(levels, j) * basis(levels, k).dag())  # j->k跃迁
        L.append(np.sqrt(gamma[j,k]) * basis(levels, k) * basis(levels, j).dag())  # k->j跃迁

# 求解Lindblad主方程
# result = mesolve(H, psi0, tlist, L, [])


result0 = mesolve(H, psi0, tlist, L ,[])


# 求解half-pi脉冲
Pulse_Ope = []
for n in range(len(tlist)):
    Ope = ((-1j * H * tlist[n]).expm()) * (HalfPi_Sy10.expm()) * ((+1j * H * tlist[n]).expm())
    state = Ope * result0.states[n]
    Pulse_Ope.append(fidelity(basis(3,1),state))
               

# 绘制各能级占据概率随时间的变化
plt.figure(figsize=(8, 5))
plt.plot(tlist, Pulse_Ope, label=f"Level")
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()
