import csv
import os

# 图片文件夹路径
image_folder = './train2'

# 获取图片文件夹中的所有图片文件路径
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if
               os.path.isfile(os.path.join(image_folder, f))]

# 指定CSV文件路径和名称
csv_filename = 'train_data2.csv'

# 打开CSV文件并写入数据
with open(csv_filename, 'w', newline='') as file:
    writer = csv.writer(file)

    # 写入标题行
    writer.writerow(['图片名称', '路径'])

    # 写入图片文件名称和路径
    for image_path in image_files:
        image_name = os.path.basename(image_path)
        writer.writerow([image_name, image_path])

print(f'CSV文件 {csv_filename} 生成完成！')