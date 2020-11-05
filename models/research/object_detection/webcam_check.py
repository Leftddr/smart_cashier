import cv2

cam = cv2.VideoCapture(0)
while True:
    key = input("캡처 숫자를 누르세요 : ")
    for num in range(0, int(key)):
        ret_val, image = cam.read()

        if ret_val:
            cv2.imshow('my webcam', image)
cv2.destroyAllWindows()