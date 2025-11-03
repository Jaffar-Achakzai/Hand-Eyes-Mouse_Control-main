import cv2
import mediapipe as mp
import pyautogui as pg
import numpy as np
import streamlit as st
from cvzone.HandTrackingModule import HandDetector
import google.generativeai as genai
from PIL import Image

# Configure Gemini API
genai.configure(api_key="AIzaSyBebcAjKepIE1jwiw13kpBVykOO_opbwNU")
model = genai.GenerativeModel('gemini-1.5-flash')

# Streamlit UI setup
st.set_page_config(layout="wide")
# st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>AI-Based Control System</h1>", unsafe_allow_html=True)
option = st.sidebar.radio("Choose an option:", ["Hand-Controlled Mouse", "Eye-Controlled Mouse", "Visual Maths"])

st.markdown(f"<h1 style='text-align: center; color: #4CAF50;'>{option}</h1>", unsafe_allow_html=True)
FRAME_WINDOW = st.empty()

# Webcam setup
cap = cv2.VideoCapture(0)
screen_w, screen_h = pg.size()
detector = HandDetector(maxHands=1, detectionCon=0.7, minTrackCon=0.5)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
prev_pos, canvas = None, None


def display_heading(text):
    """Display dynamic headings in a visually appealing format."""
    st.markdown(f"<h1 style='text-align: center; color: #4CAF50;'>{text}</h2>", unsafe_allow_html=True)


def hand_controlled_mouse():
    # display_heading("Hand-Controlled Mouse 🖱️")
    while cap.isOpened():
        suc, frame = cap.read()
        frame = cv2.flip(frame, 1)
        hands, img = detector.findHands(frame, draw=False)

        if hands:
            hand1 = hands[0]
            fingers = detector.fingersUp(hand1)
            lmlist = hand1["lmList"]
            x, y = lmlist[8][0], lmlist[8][1]
            screen_x, screen_y = int(screen_w / frame.shape[1] * x), int(screen_h / frame.shape[0] * y)
            cv2.circle(frame, (x, y), 10, (0, 255, 255))

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

        FRAME_WINDOW.image(frame, channels='BGR')


def eye_controlled_mouse():
    # display_heading("Eye-Controlled Mouse 👀")
    while cap.isOpened():
        suc, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame_h, frame_w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb_frame)

        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0].landmark
            x, y = int(landmarks[474].x * frame_w), int(landmarks[474].y * frame_h)
            screen_x = int(screen_w / frame_w * x)
            screen_y = int(screen_h / frame_h * y)
            cv2.circle(frame, (x, y), 10, (0, 255, 255))
            pg.moveTo(screen_x, screen_y)

            left_eye = [landmarks[145], landmarks[159]]
            if abs(left_eye[0].y - left_eye[1].y) < 0.005:
                pg.click()
                pg.sleep(1)

        FRAME_WINDOW.image(frame, channels='BGR')


def visual_maths():
    # display_heading("Visual Maths ✏️")
    global prev_pos, canvas

    # col1, col2 = st.columns(2)
    col1, col2 = st.columns([3, 2])
    with col1:
        # run = st.checkbox('Run', value=True)
        FRAME_WINDOW = st.image([])

    with col2:
        output_text = st.title('🧮 Response:')
        answer = st.subheader('')

    while cap.isOpened():
        suc, img = cap.read()
        img = cv2.flip(img, 1)
        if canvas is None:
            canvas = np.zeros_like(img)
        hands, img = detector.findHands(img, draw=False)

        if hands:
            fingers, lmlist = detector.fingersUp(hands[0]), hands[0]["lmList"]
            if fingers == [0, 1, 0, 0, 0] or fingers == [1, 1, 0, 0, 0]:
                current_pos = lmlist[8][0:2]
                if prev_pos is None:
                    prev_pos = current_pos
                cv2.line(canvas, tuple(current_pos), tuple(prev_pos), (0, 255, 255), 10)
                prev_pos = current_pos
            elif fingers == [1, 1, 1, 1, 1]:
                canvas = np.zeros_like(img)
            elif fingers == [1, 1, 1, 1, 0]:
                image = Image.fromarray(canvas)
                response = model.generate_content(["Solve this Maths problem", image])
                # st.subheader(response.text)
                # col2.header("🧮 AI Response:")
                col2.write(response.text)
            else:
                prev_pos = None


        combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
        FRAME_WINDOW.image(combined, channels='BGR')


# Run the selected function
if option == "Hand-Controlled Mouse":
    hand_controlled_mouse()
elif option == "Eye-Controlled Mouse":
    eye_controlled_mouse()
elif option == "Visual Maths":
    visual_maths()
#thankyou for your time and efforts!
