import os

from PIL import Image

path = './train'
newpath = './train2'


def image(path):
    files = os.listdir(path)
    for i in files:
        files = os.path.join(path, i)
        img = Image.open(files).convert('RGB')
        dirpath = newpath
        file_name, file_extend = os.path.splitext(i)
        dst = os.path.join(os.path.abspath(dirpath), file_name + '.jpg')
        img.save(dst)


image(path)