import cv2
import mediapipe as mp 

mp_hands = mp.solutions.hands
# setting the min detection confidence to improve accuracy
hands = mp_hands.Hands(min_detection_confidence=0.7)

# initializing MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: continue

    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            raised_fingers = 0
            
            handedness = results.multi_handedness[hand_idx].classification[0].label
            
            displayed_handedness = "Left" if handedness == "Right" else "Right"
            is_right_hand = (displayed_handedness == "Right")
            
            cv2.putText(frame, f"{displayed_handedness} Hand", (10, 30 + 60 * hand_idx), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if is_right_hand:
                if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x:
                    raised_fingers += 1
            else:
                if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x:
                    raised_fingers += 1

            if (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < 
                hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y):
                raised_fingers += 1
                
            if (hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < 
                hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y):
                raised_fingers += 1
                
            if (hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < 
                hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y):
                raised_fingers += 1
                
            if (hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < 
                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y):
                raised_fingers += 1
            
            cv2.putText(frame, f"Raised Fingers: {raised_fingers}", (10, 70 + 60 * hand_idx), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Hand Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()