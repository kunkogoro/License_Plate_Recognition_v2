
import cv2
import numpy as np
from PIL import Image
import base64
import io

# class Run3(object):
#     def __init__(self):
#         self.model = E2E2()
#         self.char_info = ""
    
def processing(data):
    decode_data = base64.b64decode(data)
    np_data = np.fromstring(decode_data,np.uint8)
    image = cv2.imdecode(np_data,cv2.IMREAD_UNCHANGED)
    img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    pil_im = Image.fromarray(img_gray)

    buff = io.BytesIO()
    pil_im.save(buff,format="PNG")
    img_str = base64.b64encode(buff.getvalue())
    return str(img_str,"UTF-8")
    # image_r = cv2.imread("./test/1.jpg")
    # self.image = self.model.predict(image_r)
    # return self.model.get_license_plate()


# if __name__ == '__main__':
#     run = Run3()
#     print(run.processing())
