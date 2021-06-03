import cv2
import mediapipe as mp
import time #for frame rate
import HandTrackingModule as htm

#initialise video capture object (webcam)
cap = cv2.VideoCapture(0)
detector = htm.handDetector(draw = True, show_fps = True)

while True:
    success, img = cap.read()
    img = detector.find_Hands(img)
    landmark_List = detector.find_poisiton(img)
   
    #Show image
    cv2.imshow("Image", img)
    cv2.waitKey(1)