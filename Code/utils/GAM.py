import numpy as np
import matplotlib.pyplot as plt

# 创建数据范围从0到100的值
values = np.random.randint(0, 101, size=(10, 100))

# 创建颜色映射
cmap = plt.cm.get_cmap('jet')

# 设置图像尺寸
plt.figure(figsize=(10, 6))

# 绘制热力图
plt.imshow(values, cmap=cmap, aspect='auto')

# 添加颜色标尺
cbar = plt.colorbar()
cbar.set_label('Value')

# 保存图像
plt.savefig('./heatmap.png')

# 显示图像
plt.show()
