import cv2
import mediapipe as mp
import numpy as np
import math
from DBMonitor import DBMonitor
import os
from Webcam import WebcamCapture
import time

class HeadPoseEstimator:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

        # 3D model reference points
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -63.6, -8.5),         # Chin
            (-43.3, 32.7, -26.0),        # Left eye outer corner
            (43.3, 32.7, -26.0),         # Right eye outer corner
            (-28.9, -28.9, -24.1),       # Left mouth corner
            (28.9, -28.9, -24.1)         # Right mouth corner
        ], dtype=np.float32)

        # Camera matrix (Assuming 640x480 resolution)
        # Approximate camera internals.
        self.camera_matrix = np.array([
            [3.20688713e+03, 0.00000000e+00, 2.70261737e+02],
            [0.00000000e+00, 3.24445028e+03, 2.44411237e+02],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ], dtype="double")
        
        self.dist_coeffs = np.zeros((4, 1))  # No lens distortion

        # Circular motion tracking
        self.last_yaw = None
        self.last_pitch = None
        self.path = []  # Stores sequence of movements
        self.circular_rotations_cw = 0
        self.circular_rotations_ccw = 0

    def detect_landmarks(self, image):
        """Detect face landmarks and return them."""
        h, w, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                image_points = np.array([
                    (face_landmarks.landmark[1].x * w, face_landmarks.landmark[1].y * h),   # Nose tip
                    (face_landmarks.landmark[152].x * w, face_landmarks.landmark[152].y * h), # Chin
                    (face_landmarks.landmark[33].x * w, face_landmarks.landmark[33].y * h),   # Left eye
                    (face_landmarks.landmark[263].x * w, face_landmarks.landmark[263].y * h), # Right eye
                    (face_landmarks.landmark[61].x * w, face_landmarks.landmark[61].y * h),   # Left mouth
                    (face_landmarks.landmark[291].x * w, face_landmarks.landmark[291].y * h)  # Right mouth
                ], dtype=np.float32)
                return image_points, face_landmarks
        return None, None

    def estimate_head_pose(self, frame):
        """Estimate head pose (yaw, pitch, roll) without using RQDecomp3x3."""
        # Solve for the rotation and translation vectors using solvePnP.
        image_points, face_landmarks = self.detect_landmarks(frame)

        try:
            success_pnp, rotation_vector, translation_vector = cv2.solvePnP(
                self.model_points,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
        except cv2.error as e:
            return None, None, None

        if success_pnp:
            # Convert rotation vector to rotation matrix.
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

            # Compute Euler angles from the rotation matrix.
            sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
            singular = sy < 1e-6
            if not singular:
                x_angle = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                y_angle = np.arctan2(-rotation_matrix[2, 0], sy)
                z_angle = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            else:
                x_angle = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
                y_angle = np.arctan2(-rotation_matrix[2, 0], sy)
                z_angle = 0
                print('kuku')

            # Convert radians to degrees.
            pitch = np.degrees(x_angle)
            if pitch > 0:
                pitch = -(pitch-180)
            else:
                pitch = -(pitch+180)
            yaw = np.degrees(y_angle)
            roll = np.degrees(z_angle)

        return yaw, pitch, roll

if __name__ == "__main__":
    # Read the database directory from the first line of the config file
    with open('config.txt', 'r') as file:
        db_dir = file.readline().strip()  # Read the first line and remove any trailing whitespace
    db_path = os.path.join(db_dir,r'SparkSync.bytes')
    
    #create instance of the SparkHeadRotation class
    hpe = HeadPoseEstimator()
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
                
                angles = hpe.estimate_head_pose(_frame)

                if angles:
                    yaw, pitch, roll = angles
                    if yaw != None and pitch != None and roll != None:
                        # Display angles
                        cv2.putText(_frame, f"Yaw: {yaw:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                        cv2.putText(_frame, f"Pitch: {pitch:.2f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                        cv2.putText(_frame, f"Roll: {roll:.2f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                cv2.imshow("Head Pose Estimation", _frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break
    except:
        print("failed to process Head Rotation")
    finally:
        webcam.stop()
        webcam.release()
        cv2.destroyAllWindows()
