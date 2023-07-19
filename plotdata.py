import os
import matplotlib.pyplot as plt

# 指定包含DPT文件的目录
directory = './data3'

# 设置Matplotlib风格样式为无格子背景
plt.style.use('default')
plt.rcParams['axes.grid'] = False

# 创建保存图片的新文件夹
output_directory = './new'
os.makedirs(output_directory, exist_ok=True)

# 遍历目录中的每个DPT文件
for filename in os.listdir(directory):
    if filename.endswith('.dpt'):
        filepath = os.path.join(directory, filename)

        # 读取DPT文件中的坐标数据
        with open(filepath, 'r') as file:
            lines = file.readlines()
            wavenumber = []
            transmittance = []
            for line in lines:
                data = line.split()  # 假设每行数据以空格分隔
                if 500 < float(data[0]) < 4000:
                    wavenumber.append(float(data[0]))
                    transmittance.append(float(data[1]))

        # 绘制坐标数据
        plt.figure()
        plt.plot(wavenumber, transmittance, linewidth=0.8)
        # plt.title(filename)  # 使用文件名作为标题
        plt.axis('off')  # 隐藏坐标轴
        plt.gca().invert_xaxis()  # 左右翻转x轴

        # 设置边距，仅包含数据部分
        plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95)

        # 保存绘图为PNG文件，并指定保存路径为新文件夹中
        output_filename = os.path.join(output_directory, filename + '.png')
        plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)

        plt.close()
