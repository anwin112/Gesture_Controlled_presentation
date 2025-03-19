import streamlit as st
import cv2
import mediapipe as mp
import time
import numpy as np
import os
import subprocess

# Streamlit UI
st.header("📽️ Presentation Assistant")
st.subheader("Deliver presentations with hand gestures.")

# Upload PowerPoint File
ppt = st.file_uploader("📂 Upload your PowerPoint file:", type=['pptx'])

# Manage session state
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

    # Save the uploaded file
    with open(save_path, "wb") as f:
        f.write(ppt.getbuffer())

    st.success("✅ Presentation uploaded successfully!")

    # Start PowerPoint slideshow
    if st.button("🚀 Start Presentation"):
        if os.name == "nt":  # Windows
            subprocess.Popen(["start", "powerpnt", "/S", save_path], shell=True)
        else:
            st.warning("⚠️ PowerPoint automation is not supported on Railway.")
        st.session_state.running = True
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
    
    # If camera cannot be opened, exit gracefully
    if not cap.isOpened():
        st.error("❌ Unable to access webcam. Railway does not support direct camera access.")
        st.session_state.running = False

    swipe_cooldown = time.time()

    def get_distance(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Failed to access webcam.")
            st.session_state.running = False
            break

        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get landmark positions
                thumb_tip = (int(hand_landmarks.landmark[4].x * frame.shape[1]), int(hand_landmarks.landmark[4].y * frame.shape[0]))
                index_tip = (int(hand_landmarks.landmark[8].x * frame.shape[1]), int(hand_landmarks.landmark[8].y * frame.shape[0]))
                pinky_tip = (int(hand_landmarks.landmark[20].x * frame.shape[1]), int(hand_landmarks.landmark[20].y * frame.shape[0]))
                wrist = (int(hand_landmarks.landmark[0].x * frame.shape[1]), int(hand_landmarks.landmark[0].y * frame.shape[0]))

                # Dynamic threshold based on hand size
                dynamic_threshold = get_distance(wrist, index_tip) * 0.3

                # Gesture: Thumbs Down to Exit
                is_thumbs_down = thumb_tip[1] > wrist[1] and get_distance(thumb_tip, wrist) > dynamic_threshold * 1.5
                if is_thumbs_down:
                    st.session_state.running = False
                    st.warning("❌ Presentation stopped by thumbs-down gesture.")
                    break

                # Gesture: Pinch to Next Slide
                if (time.time() - swipe_cooldown) > 1:
                    if get_distance(thumb_tip, index_tip) < dynamic_threshold:
                        if os.name == "nt":  # Windows only
                            subprocess.run(["powershell", "-command", "(New-Object -ComObject WScript.Shell).SendKeys('{RIGHT}')"])
                        st.warning("➡️ Next Slide")
                        swipe_cooldown = time.time()

                    # Gesture: Pinch with Pinky for Previous Slide
                    elif get_distance(thumb_tip, pinky_tip) < dynamic_threshold:
                        if os.name == "nt":  # Windows only
                            subprocess.run(["powershell", "-command", "(New-Object -ComObject WScript.Shell).SendKeys('{LEFT}')"])
                        st.warning("⬅️ Previous Slide")
                        swipe_cooldown = time.time()

        # Display frame in Streamlit (instead of cv2.imshow)
        st.image(img_rgb, channels="RGB", use_column_width=True)

    cap.release()
