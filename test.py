'''
Description: 
version: 
Author: Hubery-Lee
E-mail: hrbeulh@126.com
Date: 2022-10-30 20:19:04
LastEditTime: 2022-10-30 22:34:34
LastEditors: Hubery-Lee
'''
'''
绘制两个电荷形成的电场
1)异种电荷
2)同种电荷
'''
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm  
# colormap 


def f(x,y):
    return x**2+y**2

x = np.linspace(-10,10,200)
y = np.linspace(-10,10,200)

X,Y =np.meshgrid(x,y)

X0 = -2
X1 = +2
for i in range(2):   # 0: 异种电荷，1：同种电荷
    F = abs((X-X0)/np.sqrt((X-X0)**2+Y**2)+(-1)**i*(X-X1)/np.sqrt((X-X1)**2+Y**2))
    E = -abs(1/np.sqrt((X-X0)**2+Y**2)+(-1)**i*1/np.sqrt((X-X1)**2+Y**2))

    plt.contourf(X, Y, F, 100, cmap='rainbow')  # surf  100 等高线层数
    plt.colorbar()
    plt.contour(X, Y, F, 10, colors='black')  # 10 等高线条数
    plt.contour(X, Y, E, levels=[-0.5, -0.4, -0.3, -0.2, -0.1, -0.05], colors='black')  # 10 等高线条数
    # plt.savefig("electron_"+str(i)+'.png')
    # plt.close()
    plt.show()