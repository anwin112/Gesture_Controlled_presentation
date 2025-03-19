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
        # Open PowerPoint and start the slideshow
        if os.name == "nt":  # Windows
            subprocess.Popen(["start", "powerpnt", "/S", save_path], shell=True)
        else:
            st.warning("PowerPoint automation is not supported in this environment.")
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
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    screen_width = int(cap.get(3))
    screen_height = int(cap.get(4))

    last_thumb_x = None  # Track previous thumb position for swipe detection
    swipe_cooldown = time.time()

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror the image
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                thumb_tip = (int(hand_landmarks.landmark[4].x * screen_width), int(hand_landmarks.landmark[4].y * screen_height))
                wrist = (int(hand_landmarks.landmark[0].x * screen_width), int(hand_landmarks.landmark[0].y * screen_height))

                # Gesture: Thumbs Down to Exit
                if thumb_tip[1] > wrist[1]:  # Thumb below wrist
                    st.session_state.running = False
                    st.warning("❌ Presentation stopped by thumbs-down gesture.")
                    break

                # Gesture: Swipe to Change Slides
                if last_thumb_x is not None:
                    movement = thumb_tip[0] - last_thumb_x  # Change in x-position
                    swipe_threshold = screen_width * 0.2  # 20% of screen width for a valid swipe

                    if (time.time() - swipe_cooldown) > 1:  # Cooldown to avoid multiple triggers
                        if movement < -swipe_threshold:  # Left Swipe → Next Slide
                            # Add a custom action for next slide
                            swipe_cooldown = time.time()

                        elif movement > swipe_threshold:  # Right Swipe → Previous Slide
                            # Add a custom action for previous slide
                            swipe_cooldown = time.time()

                last_thumb_x = thumb_tip[0]  # Store current thumb position for next frame

                # Draw hand landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Gesture-Controlled Slideshow", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            st.session_state.running = False
            break

    cap.release()
    cv2.destroyAllWindows()
