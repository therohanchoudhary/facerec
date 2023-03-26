import cv2
import face_recognition
import os

imgs = ['amitabh', 'amitabhyoung', 'cr7', 'messi']

for i in imgs:
    img = cv2.imread("./images/" + i + "1.jpeg")
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_encoding = face_recognition.face_encodings(rgb_img)[0]

    directory = './images/'

    res_list = []

    for f in os.listdir(directory):
        filename = os.path.join(directory, f)
        img2 = cv2.imread(filename)
        rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        for face_enc in face_recognition.face_encodings(rgb_img2):
            img_encoding2 = face_enc
            result = face_recognition.compare_faces([img_encoding], img_encoding2)
            if result[0]:
                res_list.append(filename)
                break

    print("Result: ", i, res_list)
