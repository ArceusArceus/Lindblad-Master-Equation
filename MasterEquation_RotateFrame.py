import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# 系统参数
levels = 3  # 能级数
S = (levels-1)/2  # 自旋
gamma = np.array([[0,98,130],[98,0,100],[130,100,0]])  # 退极化率

# 定义哈密顿量
H = qzero(levels)

Sy10 = Qobj([[0,-1j * 0.5,0],[1j * 0.5,0,0],[0,0,1]]) # 0，+1的两能级Sy
HalfPi_Sy10 = (-1j * 0.5 * np.pi * Sy10).expm() # Sy的pi/2脉冲



# 定义初始状态（假设实验操控+1和0能级）
psi0 = Qobj([[1],[1],[0]]) / np.sqrt(2)  # 初态为0，+1的两能级+x态


# 定义时间点
tlist = np.linspace(0, 10 ** (-3), 10000) # 这里时间的增加要达到ns量级


# 定义Lindblad算符
L = []
for j in range(levels):
    for k in range(j + 1, levels):
        L.append(np.sqrt(gamma[j,k]) * basis(levels, j) * basis(levels, k).dag())  # j->k跃迁
        L.append(np.sqrt(gamma[j,k]) * basis(levels, k) * basis(levels, j).dag())  # k->j跃迁

# 求解Lindblad主方程

result = mesolve(H, psi0, tlist, L, [])


# 施加旋波half-pi脉冲，并计算0态概率
Coherence = []
Proj = basis(3,1) * basis(3,1).dag()
for n in range(len(tlist)):
    density = result.states[n]
    density_Rot = HalfPi_Sy10 * density * HalfPi_Sy10.dag()
    Coherence.append((density_Rot * Proj).tr())

               

# 绘制0能级占据概率随时间的变化
plt.figure(figsize=(8, 5))
plt.plot(tlist, Coherence, label=f"Level")
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()
