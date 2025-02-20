import cv2
import mediapipe as mp
from DBMonitor import DBMonitor
import os
from Webcam import WebcamCapture
import numpy as np
from utils import utils
import time

class SparkHeadRotation:
    def __init__(self, debug=False):
        self._IS_DEBUG = debug
        
        self.YawBasePosture = 0
        self.PitchBasePosture = 0
        self.BasePostureCounter = 0
        self.yaw_count = 0
        self.pitch_count = 0
        self.head_rotation_count = 0
        self.yaw_crossed = False
        self.pitch_crossed = False

        # Initialize Mediapipe Pose and drawing utilities
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # Thresholds for counting rotations
        self.YAW_THRESHOLD = 35   # Left/Right rotation in pixles
        self.PITCH_THRESHOLD = 35  # Up/Down rotation in pixles
    
    def calculate_head_rotation(self, landmarks, h, w):
        """
        Calculate yaw (left-right) and pitch (up-down) displacement of head.
        Yaw is based on the horizontal movement of the nose relative to shoulders midpoint.
        Pitch is based on the vertical movement of the nose relative to shoulders midpoint.
        """
        
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value] #nose location in the frame (between 0 and 1 where 0 is left, 1 is right)
        nose_pixles = utils.coordinates_to_pixles(w,h,nose.x,nose.y) #nose position in the frame (in pixels)

        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value] #left shoulder location in the frame (0 is left, 1 is right)
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value] #right shoulder location in the frame (0 is left, 1 is right)
        
        # Midpoint of shoulders (0 is left, 1 is right)
        shoulder_midpoint = np.array([(left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2])
        # Midpoint of shoulders in pixles
        shoulder_midpoint_pixles = utils.coordinates_to_pixles(w,h,shoulder_midpoint[0],shoulder_midpoint[1])

        # Yaw: horizontal displacement of the nose relative to shoulder midpoint
        yaw = abs(nose_pixles[0] - shoulder_midpoint_pixles[0])

        # Pitch: vertical displacement of the nose relative to shoulder midpoint
        pitch = abs(nose_pixles[1] - shoulder_midpoint_pixles[1])

        return yaw, pitch
    def process(self, frame):
        # Process the image and extract landmarks
        result = self.pose.process(frame)
        h, w, _ = frame.shape

        if result.pose_landmarks:
            # Draw pose landmarks on the frame
            if self._IS_DEBUG:
                self.mp_drawing.draw_landmarks(
                    frame, result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            # Calculate yaw and pitch angles
            landmarks = result.pose_landmarks.landmark
            yaw, pitch = self.calculate_head_rotation(landmarks, h, w)

            if self.BasePostureCounter < 100:
                self.YawBasePosture = self.YawBasePosture*0.2+yaw*0.8
                self.PitchBasePosture = self.PitchBasePosture*0.2+pitch*0.8
                self.BasePostureCounter = self.BasePostureCounter+1
                if self._IS_DEBUG:
                    print(f"yaw: {yaw}")
                    print(f"pitch: {pitch}")
                    print(f"YawbasePosture: {self.YawBasePosture}")
                    print(f"PitchBasePosture: {self.PitchBasePosture}")
                    print(f"BasePostureCounter: {self.BasePostureCounter}")
            else:
                # Count yaw rotations (left-right)
                if abs(self.YawBasePosture-yaw) > self.YAW_THRESHOLD and not self.yaw_crossed:
                    self.yaw_count += 1
                    self.yaw_crossed = True
                elif abs(self.YawBasePosture-yaw) <= self.YAW_THRESHOLD:
                    self.yaw_crossed = False

                # Count pitch rotations (up-down)
                if abs(self.PitchBasePosture-pitch) > self.PITCH_THRESHOLD and not self.pitch_crossed:
                    self.pitch_count += 1
                    self.pitch_crossed = True
                elif abs(self.PitchBasePosture-pitch) <= self.PITCH_THRESHOLD:
                    self.pitch_crossed = False
            
            if self.pitch_count > 1 and self.yaw_count > 1:
                self.pitch_count = 0
                self.yaw_count = 0
                self.head_rotation_count = self.head_rotation_count + 1
            
            if self._IS_DEBUG:
                # Display angles and counts
                cv2.putText(frame, f'Yaw (L-R): {yaw}', (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, f'Pitch (U-D): {pitch}', (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, f'Base Yaw: {int(self.YawBasePosture)}', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, f'Base Pitch: {int(self.PitchBasePosture)}', (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, f'Yaw Rotations: {self.yaw_count}', (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, f'Pitch Rotations: {self.pitch_count}', (10, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, f'Rotations: {self.head_rotation_count} ', (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
                    
        return (frame, self.head_rotation_count)
    
    def is_patient_in_place(self, frame):
        # Process the image and extract landmarks
        result = self.pose.process(frame)
        h, w, _ = frame.shape

        

if __name__ == "__main__":
    # Read the database directory from the first line of the config file
    with open('config.txt', 'r') as file:
        db_dir = file.readline().strip()  # Read the first line and remove any trailing whitespace
    db_path = os.path.join(db_dir,r'SparkSync.bytes')
    
    #create instance of the SparkHeadRotation class
    hr = SparkHeadRotation(True)
    #create webcam capture manager
    webcam = WebcamCapture()
    #initilize DB monitor
    dm = DBMonitor(db_path)
    
    try:
        webcam.init()
        webcam.start()
        time.sleep(2.5)
        while True:
            _frame = webcam.get_frame()
            if _frame is not None:
                processed_frame, head_rotations = hr.process(_frame)
                dm.set_HeadRotations(head_rotations)
                cv2.imshow("Pupil Connection", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break
    except:
        print("failed to process Head Rotation")
    finally:
        webcam.stop()
        webcam.release()
        cv2.destroyAllWindows()