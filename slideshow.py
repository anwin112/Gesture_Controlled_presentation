import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

# Start webcam capture
cap = cv2.VideoCapture(0)
screen_width = int(cap.get(3))
screen_height = int(cap.get(4))

swipe_cooldown = time.time()  # Cooldown to prevent multiple triggers
gesture_detected = None  # Track last detected gesture
gesture_time = time.time()  # Track gesture hold time
pointer_active = False  # Pointer mode flag
pointer_start_time = None  # Time when pointer was first activated

def get_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip horizontally for natural movement
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get fingertip coordinates
            thumb_tip = (int(hand_landmarks.landmark[4].x * screen_width), int(hand_landmarks.landmark[4].y * screen_height))
            index_tip = (int(hand_landmarks.landmark[8].x * screen_width), int(hand_landmarks.landmark[8].y * screen_height))
            middle_tip = (int(hand_landmarks.landmark[12].x * screen_width), int(hand_landmarks.landmark[12].y * screen_height))
            ring_tip = (int(hand_landmarks.landmark[16].x * screen_width), int(hand_landmarks.landmark[16].y * screen_height))
            pinky_tip = (int(hand_landmarks.landmark[20].x * screen_width), int(hand_landmarks.landmark[20].y * screen_height))
            wrist = (int(hand_landmarks.landmark[0].x * screen_width), int(hand_landmarks.landmark[0].y * screen_height))

            # Dynamic threshold based on hand size
            dynamic_threshold = get_distance(wrist, index_tip) * 0.3  # 30% of hand size

            # Draw fingertips
            cv2.circle(frame, thumb_tip, 10, (0, 0, 255), -1)  # Red - Thumb
            cv2.circle(frame, index_tip, 10, (0, 255, 0), -1)  # Green - Index
            cv2.circle(frame, pinky_tip, 10, (255, 0, 0), -1)  # Blue - Pinky

            # Detect "Thumbs Down" Gesture to Exit
            is_thumbs_down = thumb_tip[1] > wrist[1] and get_distance(thumb_tip, wrist) > dynamic_threshold * 1.5

            if is_thumbs_down:
                print("Exiting Slideshow...")
                pyautogui.press("esc")  # Exit slideshow
                break  # Exit loop

            # Gesture-based Slide Control (1-second cooldown)
            if (time.time() - swipe_cooldown) > 1:
                if get_distance(thumb_tip, index_tip) < dynamic_threshold:
                    if gesture_detected == "NEXT" and (time.time() - gesture_time) < 0.05:
                        continue
                    pyautogui.press("right")
                    print("Next Slide -->")
                    swipe_cooldown = time.time()
                    gesture_detected = "NEXT"
                    gesture_time = time.time()
                
                elif get_distance(thumb_tip, pinky_tip) < dynamic_threshold:
                    if gesture_detected == "PREV" and (time.time() - gesture_time) < 0.05:
                        continue
                    pyautogui.press("left")
                    print("Previous Slide <--")
                    swipe_cooldown = time.time()
                    gesture_detected = "PREV"
                    gesture_time = time.time()

            # **Pointer Mode (Laser Pointer)**
            if pointer_active:
                # Move mouse pointer to index finger position
                screen_x = int(index_tip[0] * (1920 / screen_width))  # Scale to 1080p
                screen_y = int(index_tip[1] * (1080 / screen_height))
                pyautogui.moveTo(screen_x, screen_y)

                # Draw "Laser Pointer" Effect
                cv2.circle(frame, index_tip, 15, (0, 0, 255), 3)  # Red Laser

            # If index finger is held up for more than 2 seconds, activate pointer mode
            if index_tip[1] < wrist[1]:  # Index finger pointing up
                if pointer_start_time is None:
                    pointer_start_time = time.time()  # Start timer
                elif time.time() - pointer_start_time > 2:  # Activate after 2 seconds
                    pointer_active = True
                    print("Pointer Mode Activated qq")
            else:
                pointer_start_time = None  # Reset timer
                pointer_active = False  # Deactivate if index is lowered

            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Gesture-Controlled Slideshow", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
