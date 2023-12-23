import matplotlib.pyplot as plt
import numpy as np

x_ucf=np.array([0.843,2.172,6.493,24.719,28.653,30.994])
y_ucf=np.array([82.44,84.67,86.97,84.30,86.98,82.30])
txt_ucf=np.array(['Wu et al.(2020)','Cao et al.(2022)','UR-DMU(2023)','RTFM(2021)','MGFN(2022)','MIST(2021)'])
maker=['o','s','d','v','x','+']
color =['b','g','k','c','m','y']
ours_x=[0.615]
ours_auc=[85.21]
plt.grid()
plt.scatter(ours_x,ours_auc,c='r',marker='*',s=90,label='Ours')

for a, b, c ,d ,e in zip (x_ucf,y_ucf,txt_ucf,maker,color):
    plt.scatter(a,b,c=e, marker=d, s=80, label=c)

plt.legend()  # 显示上面的label
plt.xlabel('Parameter(M)',fontsize=18)
plt.ylabel('AUC(%)',fontsize=18)  # accuracy
plt.ylim(72,88)
plt.title("UCF-Crime",fontsize=18)
# plt.xlim(0,35)
# plt.xlim(0,1)#仅设置y轴坐标范围
# plt.show()
plt.savefig('param_ucf.png',dpi=300)
