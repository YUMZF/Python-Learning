# _*_ coding: utf-8 _*_
# Author: YUM
# Creation Date: 2018/12/18

# todo scatter
# import matplotlib.pyplot as plt
# import numpy as np
#
# n = 1024
#
# X = np.random.normal(0, 1, n)
# Y = np.random.normal(0, 1, n)
# T = np.arctan2(Y, X)  # for color value
#
# plt.scatter(X, Y, s=75, c=T, alpha=0.5)
#
# plt.xlim((-1.5, 1.5))
# plt.ylim((-1.5, 1.5))
#
# plt.xticks(())  # 取消坐标轴数据的显示
# plt.yticks(())
#
# plt.show()


# todo 等高线

# import matplotlib.pyplot as plt
# import numpy as np
#
#
# def f(x, y):
#     # the height function
#     return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)
#
#
# n = 256
# x = np.linspace(-3, 3, n)
# y = np.linspace(-3, 3, n)
# X, Y = np.meshgrid(x, y)
#
# plt.contourf(X, Y, f(X, Y), 8, alpha=0.75, cmap=plt.cm.hot)
#
# C = plt.contour(X, Y, f(X, Y), 8, colors='black')
#
# plt.clabel(C, inline=True, fontsize=10)
#
# plt.xticks(())
# plt.yticks(())
#
# plt.show()

# todo 3D

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X ** 2 + Y ** 2)
Z = np.sin(R)

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))

ax.contourf(X,Y,Z,zdir='z',offset=-2,cmap='rainbow')
ax.set_zlim(-2,2)
plt.show()


# todo subplot 1  调整figure
# import matplotlib.pyplot as plt

# plt.figure()
#
# plt.subplot(2, 1, 1)  # plt.subplot(2,2,2)2行2列的第1个位置
# plt.plot([0, 1], [0, 1])
#
# plt.subplot(2, 3, 4)
# plt.plot([0, 1], [0, 2])
#
# plt.subplot(235)        # 该种表达方式也是可行的
# plt.plot([0,1],[0,3])
#
# plt.subplot(236)
# plt.plot([0,1],[0,4])
#
# plt.show()

# todo subplot 2
# import matplotlib.pyplot as plt
#
# import matplotlib.gridspec as gridspec
#
# # method 1:subplot2grid
# #
# # plt.figure()
# # ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3, rowspan=1)
# # ax1.plot([1, 2], [1, 2])
# # ax1.set_title('ax1_title')
# # ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
# # ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
# # ax4 = plt.subplot2grid((3, 3), (2, 0))
# # ax5 = plt.subplot2grid((3, 3), (2, 1))
#
#
# # method 2:gridspec
#
# # plt.figure()
# # gs = gridspec.GridSpec(3, 3)
# # ax1 = plt.subplot(gs[0, :])     # 第0行占了之后所有的列
# # ax2 = plt.subplot(gs[1, :2])    # 第1行占了2列
# # ax3 = plt.subplot(gs[1:, 2])    # 第1行之后占2行
# # ax4 = plt.subplot(gs[-1, 0])
# # ax5 = plt.subplot(gs[-1, -2])
# #
# # ax1.plot([0,1],[0,1])
# # ax2.plot([0,2],[0,2])
# # ax3.plot([0,3],[0,3])
# # ax4.plot([0,4],[0,4])
# # ax5.plot([0,5],[0,5])
#
# # method 3:easy to define structure
#
# f, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2, 3, sharex=True, sharey=True)  # sharex分享x轴
# ax11.scatter([1, 2], [1, 2])
#
# plt.tight_layout()  # 自动排版
#
# plt.show()

# todo 图中图

# import matplotlib.pyplot as plt
#
# fig = plt.figure()
# x = [1, 2, 3, 4, 5, 6, 7]
# y = [1, 3, 4, 2, 5, 8, 6]
#
# left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
# ax1 = fig.add_axes([left, bottom, width, height])       # 百分比
# ax1.plot(x,y,'r')
# ax1.set_xlabel('x')
# ax1.set_ylabel('y')
# ax1.set_title('title')
#
# left, bottom, width, height = 0.2, 0.6, 0.25, 0.25
# ax2 = fig.add_axes([left, bottom, width, height])
# ax2.plot(y,x,'b')
# ax2.set_xlabel('x')
# ax2.set_ylabel('y')
# ax2.set_title('title inside 1')
#
# plt.axes([0.6, 0.2, 0.25, 0.25])        # method 2
# plt.plot(y[::-1],x,'g')
# plt.xlabel('x')
# plt.ylabel('y')
#
# plt.title('title inside 2')
#
#
# plt.show

# todo 主次坐标轴

# import matplotlib.pyplot as plt
# import numpy as np
#
# x = np.arange(0, 10, 0.1)
# y1 = 0.05 * x ** 2
# y2 = -1 * y1
#
# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
# ax1.plot(x, y1, 'g-')
# ax2.plot(x, y2, 'b--')
#
# ax1.set_xlabel('X data')
# ax1.set_ylabel('Y1', color='g')
# ax2.set_ylabel('Y2', color='b')
#
# plt.show()

# todo Animation

# import numpy as np
# from matplotlib import pyplot as plt
# from matplotlib import animation
#
# fig, ax = plt.subplots()
#
# x = np.arange(0, 2 * np.pi, 0.01)
# line, = ax.plot(x, np.sin(x))
#
#
# def animate(i):
#     line.set_ydata(np.sin(x + i / 10))
#     return line,
#
#
# def init():
#     line.set_ydata(np.sin(x))
#     return line,
#
#
# ani = animation.FuncAnimation(fig=fig, func=animate, frames=1000, init_func=init, interval=20, blit=False)
#
# plt.show()
