#请注意：这个程序是在旋转坐标系中进行计算的
import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from scipy.optimize import curve_fit

# 系统参数
levels = 3  # 能级数
S = (levels-1)/2  # 自旋
gamma = np.array([[0,100,130],[100,0,100],[130,100,0]])  # 退极化率

# 定义哈密顿量，旋转坐标系下Lindblad方程不含H，所以设为零
H = qzero(levels)

Sy10 = Qobj([[0,-1j * 0.5,0],[1j * 0.5,0,0],[0,0,1]]) # 0，+1的两能级Sy
HalfPi_Sy10 = (-1j * 0.5 * np.pi * Sy10).expm() # Sy的pi/2脉冲



# 定义初始状态（假设实验操控+1和0能级）
psi0 = Qobj([[1],[1],[0]]) / np.sqrt(2)  # 初态为0，+1的两能级+x态

# 定义时间点
tlist = np.linspace(0, 35 * 10 ** (-3), 100000) 


# 定义Lindblad算符
L = []
for j in range(levels):
    for k in range(j + 1, levels):
        L.append(np.sqrt(gamma[j,k]) * basis(levels, j) * basis(levels, k).dag())  # j->k跃迁
        L.append(np.sqrt(gamma[j,k]) * basis(levels, k) * basis(levels, j).dag())  # k->j跃迁

# 求解Lindblad主方程

result = mesolve(H, psi0, tlist, L, [])


# 施加half-pi脉冲，并计算0态概率
# Coherence = []
# Proj = basis(3,1) * basis(3,1).dag()
# for n in range(len(tlist)):
#     density = result.states[n]
#     density_Rot = HalfPi_Sy10 * density * HalfPi_Sy10.dag()
#     Coherence.append((density_Rot * Proj).tr())
Coherence10 = []
Coherence1_1 = []
Coherence0_1 = []
for n in range(len(tlist)):
    density = result.states[n]
    Coherence10.append(2 * np.real(density[0,1]))
    Coherence1_1.append(2 * np.real(density[0,2]))
    Coherence0_1.append(2 * np.real(density[1,2]))
               

# 绘制0能级占据概率随时间的变化
plt.figure(figsize=(8, 5))
plt.plot(tlist, Coherence10, label=f"10")
plt.plot(tlist, Coherence1_1, label=f"1-1")
plt.plot(tlist, Coherence0_1, label=f"0-1")
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()

#拟合T2
def func(t, T2):
    return np.exp(- t / T2)

T1 = 1 / (2 * gamma[0,1] + gamma[0,2] + gamma[1,2])

popt, pcov = curve_fit(func, tlist, Coherence10)
plt.scatter(tlist, Coherence10, label='Data')
plt.plot(tlist, func(tlist, *popt), 'r-', label= f"T2 = {popt} 2T1 = {2 * T1}")
plt.legend()
plt.show()

