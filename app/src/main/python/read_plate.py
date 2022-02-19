
import cv2
import numpy as np
from lib_detection import load_model, detect_lp, im2single
from os.path import dirname, join
from tensorflow.keras.models import model_from_json
from joblib import load
import base64

# Ham sap xep contour(viền) tu trai sang phai


# Dinh nghia cac ky tu tren bien so
char_list =  '0123456789ABCDEFGHKLMNPRSTUVXYZ'

# Ham fine tune bien so, loai bo cac ki tu khong hop ly
Ivehicle = None


# Đường dẫn ảnh, các bạn đổi tên file tại đây để thử nhé
# img_path = "1.jpg"
# Ivehicle = cv2.imread(join(dirname(__file__),img_path))
# print(Ivehicle)

# Load model LP detection
wpod_net_path = join(dirname(__file__), "wpod-net_update1.json")
# wpod_net_path = "weight/wpod-net_update1.json"
json_file = open(wpod_net_path, 'r')
loaded_model_json = json_file.read()
json_file.close()
# wpod_net = model_from_json(loaded_model_json)
loaded_model_svm = model_from_json(loaded_model_json)
# wpod_net = load_model(loaded_model_json)
mModelSVM = join(dirname(__file__), "wpod-net_update1.h5")
wpod_net = load_model(mModelSVM)
# Đọc file ảnh đầu vào

# Kích thước lớn nhất và nhỏ nhất của 1 chiều ảnh
Dmax = 608
Dmin = 288

# Lấy tỷ lệ giữa W và H của ảnh và tìm ra chiều nhỏ nhất



# Cau hinh tham so cho model SVM
digit_w = 30 # Kich thuoc ki tu
digit_h = 60 # Kich thuoc ki tu

mModelSVM = join(dirname(__file__), "svmModelNew1.joblib")
model_svm = load(mModelSVM)

# model_svm = cv2.ml.SVM_load(join(dirname(__file__),'svm.xml'))

def processing(data):
    Ivehicle = getImageFromFile(data)

    ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)

    _ , LpImg, lp_type = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, lp_threshold=0.5)

    if (len(LpImg)):
        print("do day")
        # Chuyen doi anh bien so
        LpImg[0] = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))

        roi = LpImg[0]

    # Chuyen anh bien so ve gray
        gray = cv2.cvtColor( LpImg[0], cv2.COLOR_BGR2GRAY)
    # print(gray)


    # Ap dung threshold de phan tach so va nen
        binary = cv2.threshold(gray, 127, 255,
                           cv2.THRESH_BINARY_INV)[1]

    # cv2.imshow("Anh bien so sau threshold", binary)
    # cv2.imwrite("step20.png", binary)

    # cv2.waitKey()

    # Segment kí tự
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
        cont, _  = cv2.findContours(thre_mor, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)



        plate_info = ""

        for c in sort_contours(cont):
            (x, y, w, h) = cv2.boundingRect(c)
            ratio = w/h
            solidity = cv2.contourArea(c) / float(w * h)
            heightRatio = h / float(binary.shape[0])
            if 0.1< ratio< 1.0 and solidity > 0.1 and 0.35 < heightRatio < 2.0: # Chon cac contour dam bao ve ratio w/h
                if h/roi.shape[0]>=0.6: # Chon cac contour cao tu 60% bien so tro len

                    # Ve khung chu nhat quanh so
                    cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 1)

                    # Tach so va predict
                    curr_num = thre_mor[y:y+h,x:x+w]
                    curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                    _, curr_num = cv2.threshold(curr_num, 30, 255, cv2.THRESH_BINARY)
                    curr_num = np.array(curr_num,dtype=np.float32)
                    curr_num = curr_num.reshape(-1, digit_w * digit_h)

                    # Dua vao model SVM
                    result = model_svm.predict(curr_num)[0]
                    result = int(result[0, 0])

                    if result<=9: # Neu la so thi hien thi luon
                        result = str(result)
                    else: #Neu la chu thi chuyen bang ASCII
                        result = chr(result)

                    plate_info +=result
        return plate_info
def sort_contours(cnts):
    reverse = False
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),key=lambda b: b[1][i], reverse=reverse))
    return cnts
def fine_tune(lp):
    newString = ""
    for i in range(len(lp)):
        if lp[i] in char_list:
            newString += lp[i]
    return newString
def getImageFromFile(file):
    decoded_data = base64.b64decode(file)
    np_data = np.fromstring(decoded_data, np.uint8)
    Ivehicle = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)

    # height, width, depth = Ivehicle.shape
    #
    # # resizing the image to find spaces better
    # Ivehicle = cv2.resize(Ivehicle, dsize=(width * 5, height * 4),
    #                       interpolation=cv2.INTER_CUBIC)

    return Ivehicle
# print(processing())
# cv2.imshow("Cac contour tim duoc", roi)
# cv2.imwrite("step3.png", roi)
# cv2.waitKey()

# Viet bien so len anh
# cv2.putText(Ivehicle,fine_tune(plate_info),(50, 50), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 0, 255), lineType=cv2.LINE_AA)

# Hien thi anh
# print("Bien so=", plate_info)
# cv2.imshow("step4.png",Ivehicle)
# cv2.waitKey()



# cv2.destroyAllWindows()
