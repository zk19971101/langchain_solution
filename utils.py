import fitz
from PIL import Image


def show_image(file_path, page_num):
    doc = fitz.open(file_path)
    # 根据页面获取当页的内容
    page = doc[page_num]
    # 将页面渲染为分辨率为300 DPI的PNG图像，从默认的72DPI转换到300DPI
    picture = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
    # 从渲染的像素数据创建一个Image对象
    image = Image.frombytes("RGB", [picture.width, picture.height], picture.samples)
    # 返回渲染后的图像
    return image