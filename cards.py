import cv2
import numpy as np

def detect_rectangle_webcam():
    
    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        blurred = cv2.GaussianBlur(gray, (5, 5), 0)


        edges = cv2.Canny(blurred, 50, 150)


        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                if 0.9 <= aspect_ratio <= 1.1:
                    cv2.drawContours(frame, [approx], 0, (0, 255, 0), 3)


        cv2.imshow('Rectangle Detection', frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()


detect_rectangle_webcam()
