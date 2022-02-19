import numpy as np
import cv2
from nhandienCNN import E2ECNN
import base64

# class Run1(object):
#     def __init__(self):
#         self.model = E2ECNN()
    
# def processing(data):
#     decode_data = base64.b64decode(data)
#     np_data = np.fromstring(decode_data,np.uint8)
#     image1 = cv2.imdecode(np_data,cv2.IMREAD_UNCHANGED)
#     image_r = cv2.imread(image1)
#     model = E2E2()
#     image = model.predict(image1)
#     return model.get_license_plate()

    # def processing(self):
    #     image_r = cv2.imread("test/1.jpg")
    #     self.model.predict(image_r)
    #     return self.model.get_license_plate()

def processing(data):
    decode_data = base64.b64decode(data)
    np_data = np.fromstring(decode_data,np.uint8)
    image1 = cv2.imdecode(np_data,cv2.IMREAD_UNCHANGED)
    # image_r = cv2.imread(image1)
    model = E2ECNN()
    image = model.predict(image1)
    return model.get_license_plate()

# print(processing())

#
# if __name__ == '__main__':
#     run = Run1()
#     print(run.processing())
