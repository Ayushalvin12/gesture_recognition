import cv2
import mediapipe as mp 

# Initialize MediaPipe hands solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize MediaPipe drawing utilities to draw landmarks on the frame
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret: continue

    # Convert the frame from BGR (OpenCV's default format) to RGB (required by MediaPipe)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            raised_fingers = 0

            if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y:
                raised_fingers += 1
            if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y:
                raised_fingers += 1
            if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y:
                raised_fingers += 1
            if hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y:
                raised_fingers += 1
            if hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y:
                raised_fingers += 1
            
            cv2.putText(frame, f"Raised Fingers: {raised_fingers}", (50,50), cv2.FONT_HERSHEY_COMPLEX,1, (0,255,0), 2)

    # Display the frame with landmarks drawn and raised finger count
    cv2.imshow('Hand Recognition', frame)

    # Exit the loop if the 'x' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

# Release the webcam and close all OpenCV windows after the loop ends
cap.release()
cv2.destroyAllWindows()