import cv2
import mediapipe as mp
import pyautogui as pg
from cvzone.HandTrackingModule import HandDetector

cam = cv2.VideoCapture(0)
screen_w, screen_h = pg.size()
# Initialize the HandDetector class with the given parameters
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

# hand_detector = mp.solutions.hands.Hands()
# draw_hand = mp.solutions.drawing_utils

while True:
    suc, frame = cam.read()
    frame_h, frame_w, _ = frame.shape
    frame = cv2.flip(frame, 1)
    hands, img = detector.findHands(frame, draw=False, flipType=False)

    if hands:
        # Information for the first hand detected
        hand1 = hands[0]  # Get the first hand detected
        lmlist = hand1["lmList"]  # List of 21 landmarks for the first hand

        # Count the number of fingers up for the first hand
        fingers = detector.fingersUp(hand1)
        print(fingers)
        # print(lmlist[8]) -> [379, 170, -53]

        x = lmlist[8][0]
        y = lmlist[8][1]
        cv2.circle(frame, (x, y), 10, (0, 255, 255))

        screen_x = int(screen_w / frame_w * x)
        screen_y = int(screen_h / frame_h * y)

        if fingers == [0, 1, 0, 0, 0] or fingers == [1, 1, 0, 0, 0]:
            # lmlist[8] -> for forefinger top
            pg.moveTo(screen_x, screen_y)

        elif fingers == [0, 1, 1, 0, 0] or fingers == [1, 1, 1, 0, 0]:
            middle_x = lmlist[12][0]
            middle_y = lmlist[12][1]
            cv2.circle(frame, (middle_x, middle_y), 10, (0, 255, 255))
            print("click")
            pg.click()
            pg.sleep(0.3)

        elif fingers == [1, 1, 1, 1, 0] or fingers == [0, 1, 1, 1, 0]:
            print("Right click")
            pg.click(button='right')
            pg.sleep(0.3)


    cv2.imshow("camera", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break










# while True:
#     suc, frame = cam.read()
#     frame = cv2.flip(frame, 1)
#
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     result = hand_detector.process(rgb_frame)
#
#     if result.multi_hand_landmarks:
#         for hand in result.multi_hand_landmarks:
#             draw_hand.draw_landmarks(frame, hand)
#
#
#     cv2.imshow("Camera", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break