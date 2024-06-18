from django.shortcuts import render

from django.http import StreamingHttpResponse
from django.shortcuts import render
import cv2
import mediapipe as mp
import numpy as np

def index(request):
    return render(request, 'index.html')

def generate_frames():
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mpDraw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    colors = {"Red": (0, 0, 255), "Green": (0, 255, 0), "Blue": (255, 0, 0), "Eraser": (0, 0, 0)}
    rectangles = {"Red": (50, 0, 200, 100), "Green": (300, 0, 200, 100), "Blue": (550, 0, 200, 100), "Eraser": (800, 0, 200, 100)}

    selectedColor = None
    drawColor = (255, 0, 255)
    brushThickness = 15
    eraserThickness = 50
    xp, yp = 0, 0
    imgCanvas = np.zeros((720, 1280, 3), np.uint8)

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLandmarks in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLandmarks, mpHands.HAND_CONNECTIONS)
                lmList = [(int(lm.x * img.shape[1]), int(lm.y * img.shape[0])) for lm in handLandmarks.landmark]
                index_x, index_y = lmList[8]
                middle_x, middle_y = lmList[12]

                for color, (x, y, w, h) in rectangles.items():
                    if x < index_x < x + w and y < index_y < y + h and x < middle_x < x + w and y < middle_y < y + h:
                        drawColor = colors[color]
                        selectedColor = color

                if lmList[8][1] < lmList[6][1] and lmList[12][1] > lmList[10][1]:
                    cv2.circle(img, (index_x, index_y), brushThickness // 2, drawColor, cv2.FILLED)
                    if xp == 0 and yp == 0:
                        xp, yp = index_x, index_y
                    if drawColor == colors["Eraser"]:
                        cv2.line(img, (xp, yp), (index_x, index_y), drawColor, eraserThickness)
                        cv2.line(imgCanvas, (xp, yp), (index_x, index_y), drawColor, eraserThickness)
                    else:
                        cv2.line(img, (xp, yp), (index_x, index_y), drawColor, brushThickness)
                        cv2.line(imgCanvas, (xp, yp), (index_x, index_y), drawColor, brushThickness)
                    xp, yp = index_x, index_y
                else:
                    xp, yp = 0, 0

        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)

        for color, (x, y, w, h) in rectangles.items():
            if selectedColor == color:
                cv2.rectangle(img, (x - 10, y - 10), (x + w + 10, y + h + 10), colors[color], -1)
            else:
                cv2.rectangle(img, (x, y), (x + w, y + h), colors[color], -1)
            cv2.putText(img, color, (x + 40, y + 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', img)
        img = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')

def video_feed(request):
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

