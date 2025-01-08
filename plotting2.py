import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def draw_point(blue_cloud, red_cloud):
    # 随机采样点云
    sample_indices = np.random.choice(blue_cloud.shape[0], 3380, replace=False)
    blue_cloud = blue_cloud[sample_indices]

    sample_indices = np.random.choice(red_cloud.shape[0], 64, replace=False)#采样点数量修改
    red_cloud = red_cloud[sample_indices]

    # 计算蓝色点云到每个红色点云的距离，并找到最近的距离
    closest_distances = np.min(np.linalg.norm(blue_cloud[:, None, :] - red_cloud[None, :, :], axis=2), axis=1)

    # 归一化距离用于颜色映射
    norm = Normalize(vmin=0, vmax=np.max(closest_distances))
    colors = plt.cm.Blues_r(norm(closest_distances))  # 使用蓝到红的渐变颜色

    # 创建3D绘图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制蓝色点云，颜色基于最近红色点的距离
    ax.scatter(blue_cloud[:, 0], blue_cloud[:, 1], blue_cloud[:, 2],
               color=colors, s=1, alpha=0.6, label='Blue Point Cloud (1024, 3)')

    # 绘制红色点云
    # ax.scatter(red_cloud[:, 0], red_cloud[:, 1], red_cloud[:, 2],
    #            color='darkblue', s=20, alpha=0.9, label='Red Point Cloud (32, 3)')

    ax.set_box_aspect((np.max(blue_cloud[:, 0]), np.max(blue_cloud[:, 1]), np.max(blue_cloud[:, 2])))
    # ax.set_xlabel('X Axis')
    # ax.set_ylabel('Y Axis')
    # ax.set_zlabel('Z Axis')
    # # 去掉网格线和坐标轴
    ax.grid(False)
    ax.set_axis_off()

    # 添加标题
    ax.set_title('Visualization of Two 3D Point Clouds with Distance-Based Colors', pad=20)

    # 显示图像
    plt.show()


def plotting2(xyzs,xyzs2):
    xyzs = xyzs.cpu().numpy()
    xyzs2 = xyzs2.cpu().numpy()

    # #激励计算
    # fs1 = fs1.cpu().numpy()
    # fs2 = fs2.cpu().numpy()
    # fs3 = fs3.cpu().numpy()
    # colors1 = fs3 - fs2
    # colors2 = fs2 - fs1

    batch = xyzs.shape[0]

    for i in range(batch):
        xyz = xyzs[i]       #全部点坐标
        xyz2 = xyzs2[i]     #采样点坐标
        #绘制点云
        draw_point(xyz,xyz2)