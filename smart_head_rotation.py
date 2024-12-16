import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Pose and drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Thresholds for counting rotations
YAW_THRESHOLD = 2   # Left/Right rotation in degrees
PITCH_THRESHOLD = 2  # Up/Down rotation in degrees

def calculate_head_rotation(landmarks):
    """
    Calculate yaw (left-right) and pitch (up-down) angles of head.
    Yaw is based on the horizontal movement of the nose relative to shoulders.
    Pitch is based on the vertical movement of the nose relative to shoulders.
    """
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    # Midpoint of shoulders
    shoulder_midpoint = np.array([(left_shoulder.x + right_shoulder.x) / 2,
                                   (left_shoulder.y + right_shoulder.y) / 2])

    # Yaw: horizontal displacement of the nose relative to shoulder midpoint
    yaw = np.arctan2(nose.x - shoulder_midpoint[0], 1) * 180 / np.pi

    # Pitch: vertical displacement of the nose relative to shoulder midpoint
    pitch = np.arctan2(nose.y - shoulder_midpoint[1], 1) * 180 / np.pi

    return yaw, pitch

def main():
    # Initialize webcam and Mediapipe Pose
    cap = cv2.VideoCapture(1)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        YawBasePosture = 0
        PitchBasePosture = 0
        BasePostureCounter = 0
        yaw_count = 0
        pitch_count = 0
        head_rotation_count = 0
        yaw_crossed = False
        pitch_crossed = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Flip the frame horizontally for a selfie-view display
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the image and extract landmarks
            result = pose.process(frame_rgb)

            if result.pose_landmarks:
                # Draw pose landmarks on the frame
                mp_drawing.draw_landmarks(
                    frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

                # Calculate yaw and pitch angles
                landmarks = result.pose_landmarks.landmark
                yaw, pitch = calculate_head_rotation(landmarks)

                if BasePostureCounter < 100:
                    YawBasePosture = YawBasePosture*0.1+yaw*0.9
                    PitchBasePosture = PitchBasePosture*0.1+pitch*0.9
                    BasePostureCounter = BasePostureCounter+1
                    print(f"YawbasePosture: {YawBasePosture}")
                    print(f"PitchBasePosture: {PitchBasePosture}")
                    print(f"BasePostureCounter: {BasePostureCounter}")
                else:
                    # Count yaw rotations (left-right)
                    if abs(YawBasePosture-yaw) > YAW_THRESHOLD and not yaw_crossed:
                        yaw_count += 1
                        yaw_crossed = True
                    elif abs(YawBasePosture-yaw) <= YAW_THRESHOLD:
                        yaw_crossed = False

                    # Count pitch rotations (up-down)
                    if abs(PitchBasePosture-pitch) > PITCH_THRESHOLD and not pitch_crossed:
                        pitch_count += 1
                        pitch_crossed = True
                    elif abs(PitchBasePosture-pitch) <= PITCH_THRESHOLD:
                        pitch_crossed = False
                
                if pitch_count > 1 and yaw_count > 1:
                    pitch_count = 0
                    yaw_count = 0
                    head_rotation_count = head_rotation_count + 1
                
                # Display angles and counts
                cv2.putText(frame, f'Yaw (L-R): {int(yaw)} deg', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f'Pitch (U-D): {int(pitch)} deg', (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f'Yaw Rotations: {yaw_count}', (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f'Pitch Rotations: {pitch_count}', (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f'Rotations: {head_rotation_count} ', (10, 190),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # Show the frame
            cv2.imshow('Head Rotation Tracker', frame)

            # Exit on 'q' key
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
