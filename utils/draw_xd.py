import matplotlib.pyplot as plt
import numpy as np
x_x = [0.01, 0.02, 0.04, 0.05, 0.1,0.15 ,0.2,0.25, 0.3]
xd_y = [81.40, 82.46, 82.67, 82.50, 82.36,81.51, 81.23, 80.24,79.81]
# ucf_y = [84.49,84.29,84.85,84.95,84.99,84.80, 84.85, 85.21,85.20,84.61,84.87]

# 画图
plt.plot(x_x, xd_y,marker="^", markersize=12, c='darksalmon', linewidth=3, label='XD-Violence')
# plt.plot(x_x, ucf_y, 'b*--', alpha=0.5, linewidth=1, label='ucf-crime')
# 设置数据标签位置及大小
for a, b in zip(x_x, xd_y):
    plt.text(a, b+0.05, '%.2f'%b, ha='center', va='bottom', fontsize=8)  # ha='center', va='top'
# for a, b1 in zip(x_x, ucf_y):
#     plt.text(a, b1, str(b1), ha='center', va='bottom', fontsize=8)
plt.legend()  # 显示上面的label

plt.xlabel('Curvature',fontsize=18)
plt.ylabel('AP(%)',fontsize=18)  # accuracy

plt.ylim(79,83)#仅设置y轴坐标范围
# plt.grid()
plt.savefig('xd.png',dpi=300)
# plt.show()
