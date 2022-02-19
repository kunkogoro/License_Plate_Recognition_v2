import cv2
import numpy as np
# from skimage import measure
from imutils import perspective
import imutils
from data_utils import order_points
from detect import detectNumberPlate
from os.path import dirname, join
from joblib import load
# from model import CNN_Model

# from skimage.filters import threshold_local
# from lib_detection import load_model, detect_lp, im2single
# import time
# from joblib import loadscipy
# import pickle


ALPHA_DICT = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'K', 9: 'L', 10: 'M', 11: 'N', 12: 'P',
              13: 'R', 14: 'S', 15: 'T', 16: 'U', 17: 'V', 18: 'X', 19: 'Y', 20: 'Z', 21: '0', 22: '1', 23: '2', 24: '3',
              25: '4', 26: '5', 27: '6', 28: '7', 29: '8', 30: '9', 31: "Background"}


class E2E2(object):
    def __init__(self):
        self.image = np.empty((28, 28, 1))
        self.detectLP = detectNumberPlate()

    #   self.recogChar = CNN_Model(trainable=False).model

        # self.recogChar.load_weights('./weights/weight.h5')
      
        self.candidates = []
         # Cau hinh tham so cho model SVM
        self.digit_w = 30 # Kich thuoc ki tu
        self.digit_h = 60 # Kich thuoc ki tu
        self.plate_info = ""

        mModelSVM = join(dirname(__file__), "svmModelNew1.joblib")
        self.model_svm = load(mModelSVM)

        # mModelSVM = join(dirname(__file__), "svm1.xml")
        # self.model_svm = cv2.ml.SVM_load(mModelSVM)
        # self.model_svm = load('./weights/modelSVM.joblib')
  
        self.char_list =  '0123456789ABCDEFGHKLMNPRSTUVXYZ'
        self.model = "CNN"


    def extractLP(self):
        # mảng chứa tọa độ và các kích thước của đối tượng
        coordinates = self.detectLP.detect(self.image)

        if len(coordinates) == 0:
            ValueError('No images detected')

        for coordinate in coordinates:
            yield coordinate



    def predict(self, image):
        # Input image or frame
        self.image = image
    


        # ratio = float(max(image.shape[:2])) / min(image.shape[:2])
        # side = int(ratio * self.Dmin)
        # bound_dim = min(side, self.Dmax)
        # _ , LpImg, lp_type = detect_lp(self.wpod_net, im2single(image), bound_dim, lp_threshold=0.5)

        # LpImg[0] = cv2.convertScaleAbs(LpImg[0], alpha=(1.0))
        # cv2.imwrite("bienso.png",LpImg[0])

        # roi = LpImg[0]
        # gray = cv2.cvtColor( LpImg[0], cv2.COLOR_BGR2GRAY)
        # binary = cv2.threshold(gray, 127, 255,
        #                  cv2.THRESH_BINARY_INV)[1]
        # cv2.imshow("Anh bien so sau threshold", binary)
        # cv2.imwrite("binary.png",binary)
        

        # self.segmentation(roi,model)

        # self.license_plate = self.format()


        for coordinate in self.extractLP():     # detect license plate by yolov3
            self.candidates = []

            # chuyển từ (x_min, y_min, width, height) sang (top left, top right, bottom left, bottom right)
            pts = order_points(coordinate)

            # cắt ảnh theo 4 tọa độ truyền vào (top left, top right, bottom left, bottom right)
            LpRegion = perspective.four_point_transform(self.image, pts)


            # cv2.imwrite('step1.png', LpRegion)
            # insert(LpRegion)
          
           
            # segmentation
            self.segmentation(LpRegion)
            

            # if(model == "CNN"):
            # # recognize characters
                
            # self.recognizeChar()
                

            # format and display license plate
            self.license_plate = self.format()

            # draw labels
            # self.image = draw_labels_and_boxes(self.image, self.license_plate, coordinate)

            # cv2.imwrite("result.png",self.image)

        #cv2.imwrite('example.png', self.image)
        return self.image
    
   #Segment tách từng kí tự trên biến số xe
    def segmentation(self, LpRegion):
        # apply thresh to extracted licences plate
        #lấy ra độ sáng của ảnh
       
        # V = cv2.split(cv2.cvtColor(LpRegion, cv2.COLOR_BGR2HSV))[2]

        # # adaptive threshold
        # # để làm nổi bật những phần mà ta muốn lấy(màu đen).
        # T = threshold_local(V, 15, offset=10, method="gaussian")
        # thresh = (V > T).astype("uint8") * 255

        gray = cv2.cvtColor( LpRegion, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray, 127, 255,
                         cv2.THRESH_BINARY_INV)[1]

        # cv2.imwrite("step2_1.png", binary)

        thresh = binary
        # cv2.imwrite("step2_2.png", thresh)
        thresh = imutils.resize(thresh, width=400)
        #SVM
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        thre_mor = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel3)

        cont, _  = cv2.findContours(thre_mor, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        #chống nhiễu hạt tiêu trong ảnh
        thresh = cv2.medianBlur(thresh, 5)
       
        # if(self.model == "SVM"):
        count = 0
                
        for c in cont:
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(thresh, (x, y), (x + w, y + h), (255, 0, 0), 2)
                aspectRatio = h / float(w)
                solidity = cv2.contourArea(c) / float(w * h)
                heightRatio = h / float(LpRegion.shape[0])
                if 1.5 < aspectRatio < 3.5 and solidity > 0.1 and 0.35 < heightRatio < 3.0: # Chon cac contour dam bao ve ratio w/h
                  if h/LpRegion.shape[0]>=0.5:
                    count = count + 1
                    cv2.rectangle(thresh, (x, y), (x + w, y + h), (255, 0, 0), 1)
                        # print(count)
                    #     # Tach so va predict
                    curr_num = thre_mor[y:y+h,x:x+w]
                    curr_num = cv2.resize(curr_num, dsize=(self.digit_w, self.digit_h))
                    # cv2.imwrite("testCurret.png",curr_num)
                    _, curr_num = cv2.threshold(curr_num, 30, 255, cv2.THRESH_BINARY)
                    curr_num = np.array(curr_num,dtype=np.float32)
                    curr_num = curr_num.reshape(-1, self.digit_w * self.digit_h)

                    #SVM
                    result = self.model_svm.predict(curr_num)
                    
                    result = result[0]
                    # print(result)
                    self.candidates.append((result,(y,x)))
           
                    self.plate_info += str(result)
            # cv2.imwrite("step3.png", thresh)
    
  
        # if(self.model == "CNN"): 
        #     labels = measure.label(thresh, connectivity=2, background=0)  
        #     for label in np.unique(labels):
        #         # if this is background label, ignore it
        #         if label == 0:
        #             continue

        #         # init mask to store the location of the character candidates
        #         # tạo một mảng với kích thước bằng với lable
        #         mask = np.zeros(thresh.shape, dtype="uint8")
        #         mask[labels == label] = 255

        #         # mask1 = np.zeros(thresh.shape, dtype="uint8")
        #         # mask1[labels == label] = 255

        #         # find contours from mask
        #         #contours: Danh sách các contour có trong bức ảnh nhị phân. 
        #         # Mỗi một contour được lưu trữ dưới dạng vector các điểm
        #         #hierarchy: Danh sách các vector, chứa mối quan hệ giữa các contour.
        #         contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #         if len(contours) > 0:
        #             contour = max(contours, key=cv2.contourArea)
                
        #             # vẽ một hình chữ nhật bao quanh label đó. x,y là tọa độ góc trên cùng bên trái, w,h là chiều dài và chiều cao
        #             (x, y, w, h) = cv2.boundingRect(contour)

        #             # rule to determine characters
        #             # Do ở bước này các giá trị thu được ngoài kí tự còn có cả nhiễu do đó ta thiết lập các giá trị ngưỡng để loại bỏ nhiễu.
        #             #Ở đây ta sử dụng ngưỡng đối với ba đại lượng:
        #             #  aspect ratio(tỉ lệ rộng / dài),
        #             #  solidity(tỉ lệ diện tích phần contour bao quanh kí tự và hình chữ nhật bao quanh kí tand w < hự)
        #             #  height ratio(tỉ lệ chiều dài kí tự / chiều dài biển số xe).
        #             aspectRatio = w / float(h)
        #             solidity = cv2.contourArea(contour) / float(w * h)
        #             heightRatio = h / float(LpRegion.shape[0])
        #             if 0.1 < aspectRatio < 1.0 and solidity > 0.1 and 0.35 < heightRatio < 2.0 :
        #                 if h/LpRegion.shape[0]>=0.5:
        #                     # extract characters
        #                     cv2.rectangle(thresh, (x, y), (x + w, y + h), (255, 0, 0), 2)
        #                     candidate = np.array(mask[y:y + h, x:x + w])
                            
        #                     # chuyển các hình không phải hình vuông thành hình vuông
        #                     square_candidate = convert2Square(candidate)

        #                     square_candidate = cv2.resize(square_candidate, (28, 28), cv2.INTER_AREA)
        #                     #cv2.imwrite('./characters/' + str(y) + "_" + str(x) + ".png", cv2.resize(square_candidate, (56, 56), cv2.INTER_AREA))
        #                     square_candidate = square_candidate.reshape((28, 28, 1))
                        
        #                     self.candidates.append((square_candidate, (y, x)))

                        
         # cv2.imwrite("step3.png", thresh)
    
    def recognizeChar(self):
        # kí tự
        characters = []
        # tọa độ tương ứng
        coordinates = []

        for char, coordinate in self.candidates:
            characters.append(char)
            coordinates.append(coordinate)
            

        characters = np.array(characters)
        #Trả về các dự đoán cho một lô mẫu duy nhất.
        result = self.recogChar.predict_on_batch(characters)

        
        result_idx = np.argmax(result, axis=1)
  
    
        self.candidates = []
        for i in range(len(result_idx)):
            if result_idx[i] == 31:    # if is background or noise, ignore it
                continue
            self.candidates.append((ALPHA_DICT[result_idx[i]], coordinates[i]))

    def format(self):
        first_line = []
        second_line = []

        for candidate, coordinate in self.candidates:
            if self.candidates[0][1][0] + 40 > coordinate[0]:
                first_line.append((candidate, coordinate[1]))
            else:
                second_line.append((candidate, coordinate[1]))

        def take_second(s):
            return s[1]

        first_line = sorted(first_line, key=take_second)
        second_line = sorted(second_line, key=take_second)

        if len(second_line) == 0:  # if license plate has 1 line
            license_plate = "".join([str(ele[0]) for ele in first_line])
        else:   # if license plate has 2 lines
            license_plate = "".join([str(ele[0]) for ele in first_line]) +  "".join([str(ele[0]) for ele in second_line])

        return license_plate

    def get_license_plate(self):
        print("SVM biền số:" + self.license_plate)
        return self.license_plate

def sort_contours(cnts):

        reverse = False
        i = 0
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                            key=lambda b: b[1][i], reverse=reverse))
        return cnts

