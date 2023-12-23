import matplotlib.pyplot as plt
import numpy as np
x_xd= np.array([ 0.843, 1.876, 6.493,24.719,28.653])
y_xd = np.array([78.64,81.69,81.66,77.81,79.19])

txt_xd =np.array(['Wu et al.(2020)','Pang et al.(2021)','UR-DMU(2023)','RTFM(2021)','MGFN(2022)'])
maker=['o','s','d','v','x']
color =['b','g','y','c','m']
ours_x=[0.615]
ours_ap=[82.67]
plt.scatter(ours_x,ours_ap,c='r',marker='*',s=80,label='Ours')

for a, b, c ,d ,e in zip (x_xd,y_xd,txt_xd,maker,color):
    plt.scatter(a,b,c=e, marker=d, s=70, label=c)

plt.legend()  # 显示上面的label
plt.grid()
plt.xlabel('Parameter(M)',fontsize=18)
plt.ylabel('AP(%)',fontsize=18)  # accuracy
plt.title("XD-Violence",fontsize=18)
# plt.ylim(72,88)
# plt.xlim(0,35)
# plt.xlim(0,1)#仅设置y轴坐标范围
# plt.show()
plt.savefig('param_xd.png',dpi=300)
