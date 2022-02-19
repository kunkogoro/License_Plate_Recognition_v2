import numpy as np
import cv2
from nhandien import E2E2
import base64

# class Run2(object):
#     def __init__(self):
#         self.model = E2E2()
#         self.char_info = ""
    
def processing(data):
    decode_data = base64.b64decode(data)
    np_data = np.fromstring(decode_data,np.uint8)
    image1 = cv2.imdecode(np_data,cv2.IMREAD_UNCHANGED)
    # image_r = cv2.imread(image1)
    model = E2E2()
    image = model.predict(image1)
    return model.get_license_plate()

# print(processing())


# if __name__ == '__main__':
#     run = Run2()
#     print(run.processing())
