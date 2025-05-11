import cv2
import mediapipe as mp

# Initialize MediaPipe Hands and drawing utility
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Start video capture from webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6) as hands:

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Flip the frame horizontally for a mirror view
        frame = cv2.flip(frame, 1)

        # Convert BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image and find hands
        result = hands.process(rgb_frame)

        # If hand landmarks are found
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw hand landmarks on the frame
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get image dimensions
                h, w, _ = frame.shape

                # Fingertip landmark indexes
                fingertip_ids = [4, 8, 12, 16, 20]

                for idx in fingertip_ids:
                    # Get fingertip coordinates
                    x = int(hand_landmarks.landmark[idx].x * w)
                    y = int(hand_landmarks.landmark[idx].y * h)

                    # Draw circle at the fingertip
                    cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

                    # Optional: Label the fingertip
                    cv2.putText(frame, f'ID:{idx}', (x - 20, y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Display the result
        cv2.imshow("Fingertip Tracking", frame)

        # Exit when ESC key is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()
