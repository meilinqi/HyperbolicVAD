import matplotlib.pyplot as plt
import numpy as np
x_x = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3,0.4,0.5,0.6,0.7]
# xd_y = [81.40, 82.46, 81.97, 82.67, 82.50, 82.36, 81.23, 79.81]
ucf_y = [84.49,84.85,84.99,84.80, 84.85, 85.21,84.57,85.20,84.61,84.87]

# 画图
# plt.plot(x_x, xd_y, 'rs--', alpha=0.5, linewidth=1, label='xd-violence')  # '
plt.plot(x_x, ucf_y, marker="s", markersize=12, c='steelblue', linewidth=3, label='UCF-Crime')
# 设置数据标签位置及大小
# for a, b in zip(x_x, xd_y):
#     plt.text(a, b, str(b), ha='center', va='bottom', fontsize=8)  # ha='center', va='top'
for a, b in zip(x_x, ucf_y):
    plt.text(a, b+0.03, '%.2f'%b, ha='center', va='bottom', fontsize=8)
# for a,b in zip (x_x, ucf_y):
#     plt.axhline(b, ls='--', c='#E7DAD2',lw=0.5)
plt.legend()  # 显示上面的label
plt.xlabel('Curvature',fontsize=18)
plt.ylabel('AUC(%)',fontsize=18)  # accuracy
plt.ylim(84,85.5)#仅设置y轴坐标范围
# plt.grid()
plt.savefig('ucf.png', dpi=300)
# plt.show()
