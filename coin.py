import numpy as np
import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math

mp_drawing = mp.solutions.drawing_utils         
mp_drawing_styles = mp.solutions.drawing_styles  
mp_hands = mp.solutions.hands    


def vector_2d_angle(v1, v2):
    v1_x = v1[0]
    v1_y = v1[1]
    v2_x = v2[0]
    v2_y = v2[1]
    try:
        angle_= math.degrees(math.acos((v1_x*v2_x+v1_y*v2_y)/(((v1_x**2+v1_y**2)**0.5)*((v2_x**2+v2_y**2)**0.5))))
    except:
        angle_ = 180
    return angle_


def hand_angle(hand_):
    angle_list = []
    # thumb 
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[2][0])),(int(hand_[0][1])-int(hand_[2][1]))),
        ((int(hand_[3][0])- int(hand_[4][0])),(int(hand_[3][1])- int(hand_[4][1])))
        )
    angle_list.append(angle_)
    # index 
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])-int(hand_[6][0])),(int(hand_[0][1])- int(hand_[6][1]))),
        ((int(hand_[7][0])- int(hand_[8][0])),(int(hand_[7][1])- int(hand_[8][1])))
        )
    angle_list.append(angle_)
    # middle 
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[10][0])),(int(hand_[0][1])- int(hand_[10][1]))),
        ((int(hand_[11][0])- int(hand_[12][0])),(int(hand_[11][1])- int(hand_[12][1])))
        )
    angle_list.append(angle_)
    # ring 
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[14][0])),(int(hand_[0][1])- int(hand_[14][1]))),
        ((int(hand_[15][0])- int(hand_[16][0])),(int(hand_[15][1])- int(hand_[16][1])))
        )
    angle_list.append(angle_)
    # pink 
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[18][0])),(int(hand_[0][1])- int(hand_[18][1]))),
        ((int(hand_[19][0])- int(hand_[20][0])),(int(hand_[19][1])- int(hand_[20][1])))
        )
    angle_list.append(angle_)
    return angle_list


def hand_pos(finger_angle):
    f1 = finger_angle[0]   
    f2 = finger_angle[1]   
    f3 = finger_angle[2]   
    f4 = finger_angle[3]   
    f5 = finger_angle[4]   



    if f1>=50 and f2>=50 and f3>=50 and f4>=50 and f5>=50:
        return '0'

    elif f1<50 and f2<50 and f3<50 and f4<50 and f5<50:
        return '5'                

cap = cv2.VideoCapture(0)
w, h = 800, 600 
square_size = 200
center_x, center_y = w // 2, h // 2

x1 = center_x - square_size // 2
y1 = center_y - square_size // 2
x2 = center_x + square_size // 2
y2 = center_y + square_size // 2
 

def detect_ellipses(image_path):

    image = cv2.imread(image_path)

    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)


    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))

        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            radius = circle[2]


            cv2.ellipse(image, center, (radius, radius), 0, 0, 360, (0, 0, 255), 2)

            
            cv2.circle(image, center, 2, (0, 255, 0), 3)
            
with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, img = cap.read()
        img = cv2.resize(img, (w,h))              
        if not ret:
            print("Cannot receive frame")
            break
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
        results = hands.process(img2)               
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                finger_points = []                   
                for i in hand_landmarks.landmark:
                    x = i.x*w
                    y = i.y*h
                    finger_points.append((x,y))
                if finger_points:
                    finger_angle = hand_angle(finger_points)          
                    text = hand_pos(finger_angle) 

                    if text == '5':   
                        x, y = (w - square_size) // 2, (h - square_size) // 2
                        x_end, y_end = x + square_size, y + square_size
                        cv2.rectangle(img, (x,y), (x_end,y_end), (0, 0, 255), 2)
                        
                        roi = img[y1:y2, x1:x2]
                        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        blurred = cv2.GaussianBlur(gray_roi, (3, 3), 0)
                        circles = cv2.HoughCircles(gray_roi, cv2.HOUGH_GRADIENT, dp=1, minDist=5,param1=20, param2=30, minRadius=10, maxRadius=50)
                        if circles is not None:
                            circles = np.uint16(np.around(circles))
                            for circle in circles[0, :]:
                                    center = (circle[0], circle[1])
                                    radius = circle[2]
                                    if radius <=20:
                                        cv2.ellipse(roi, center, (radius, radius), 0, 0, 360, (0, 0, 0), -1)
                                    
        cv2.imshow("Detected Coin", roi)
        if cv2.waitKey(5) == ord('q'):
            break    
    cap.release()
    cv2.destroyAllWindows()
