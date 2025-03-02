import base64
from io import BytesIO

from PIL import Image


def get_test_image():
    image_file = Image.open("test_image.png")
    image_file = image_file.resize((320, 320))
    buffer = BytesIO()
    image_file.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
