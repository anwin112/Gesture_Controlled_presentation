import streamlit as st
import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
import os

# Streamlit UI
st.header("📽️ Presentation Assistant")
st.subheader("Deliver presentations with hand gestures.")

# Upload PowerPoint File
ppt = st.file_uploader("📂 Upload your PowerPoint file:", type=['pptx'])

# Manage session state for controlling the gesture system
if "running" not in st.session_state:
    st.session_state.running = False

if ppt:
    save_path = "temp_presentation.pptx"
    
    if os.path.exists(save_path):
        try:
            os.remove(save_path)  # Delete existing file
        except PermissionError:
            st.error("❌ Close the previous PowerPoint file before uploading a new one.")
            st.stop()

    # Save the new uploaded file
    with open(save_path, "wb") as f:
        f.write(ppt.getbuffer())

    st.success("✅ Presentation uploaded successfully!")

    # Start PowerPoint slideshow
    if st.button("🚀 Start Presentation"):
        os.startfile(save_path)
        time.sleep(3)
        pyautogui.press("f5")
        st.session_state.running = True  # Enable gesture control
        st.warning("🎤 Slideshow started! Use gestures to navigate.")

# **Stop Gesture Control Button**
if st.session_state.running:
    if st.button("🛑 Stop Gesture Control"):
        st.session_state.running = False
        st.warning("❌ Gesture control stopped!")

# **Gesture Control Code**
if st.session_state.running:
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    screen_width = int(cap.get(3))
    screen_height = int(cap.get(4))

    swipe_cooldown = time.time()
    pointer_active = False
    pointer_start_time = None

    def get_distance(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                thumb_tip = (int(hand_landmarks.landmark[4].x * screen_width), int(hand_landmarks.landmark[4].y * screen_height))
                index_tip = (int(hand_landmarks.landmark[8].x * screen_width), int(hand_landmarks.landmark[8].y * screen_height))
                pinky_tip = (int(hand_landmarks.landmark[20].x * screen_width), int(hand_landmarks.landmark[20].y * screen_height))
                wrist = (int(hand_landmarks.landmark[0].x * screen_width), int(hand_landmarks.landmark[0].y * screen_height))

                dynamic_threshold = get_distance(wrist, index_tip) * 0.3

                is_thumbs_down = thumb_tip[1] > wrist[1] and get_distance(thumb_tip, wrist) > dynamic_threshold * 1.5

                if is_thumbs_down:
                    pyautogui.press("esc")
                    st.session_state.running = False
                    break

                if (time.time() - swipe_cooldown) > 1:
                    if get_distance(thumb_tip, index_tip) < dynamic_threshold:
                        pyautogui.press("right")
                        swipe_cooldown = time.time()

                    elif get_distance(thumb_tip, pinky_tip) < dynamic_threshold:
                        pyautogui.press("left")
                        swipe_cooldown = time.time()

                if pointer_active:
                    screen_x = int(index_tip[0] * (1920 / screen_width))
                    screen_y = int(index_tip[1] * (1080 / screen_height))
                    pyautogui.moveTo(screen_x, screen_y)
                    cv2.circle(frame, index_tip, 15, (0, 0, 255), 3)

                if index_tip[1] < wrist[1]:
                    if pointer_start_time is None:
                        pointer_start_time = time.time()
                    elif time.time() - pointer_start_time > 2:
                        pointer_active = True
                else:
                    pointer_start_time = None
                    pointer_active = False

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Gesture-Controlled Slideshow", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            st.session_state.running = False
            break

    cap.release()
    cv2.destroyAllWindows()

