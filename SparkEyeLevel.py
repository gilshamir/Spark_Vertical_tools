import cv2
import mediapipe as mp
from DBMonitor import DBMonitor
import os
from Webcam import WebcamCapture
from utils import utils, Point
import numpy as np


class SparkEyeLevel:
    def __init__(self, debug=True):
        self._IS_DEBUG = debug
        self.frame = None

        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.focal_length = 8 #in mm
        self.ccd_px_size = 0.006 #in mm
        self.ccd_height_px = 1280 #
        self.screen_height_mm = 700
        self.camera_above_screen = 55
        self.ccd_ang = 26 * np.pi / 180
        self.horizontal_camera_angle = 0
        self.vertical_camera_angle = 0
        self.patient_distance = 600
        self.screen_height_px = 1920
        

        # Define indices for the pupil approximation
        #https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png
        self._RIGHT_EYE_NASAL_INDEX = 133
        self._LEFT_EYE_NASAL_INDEX = 362
        self._RIGHT_EYE_TEMPORAL_INDEX = 33
        self._LEFT_EYE_TEMPORAL_INDEX = 263

        #pupils
        self._RIGHT_EYE_PUPIL_INDEX = 159
        self._LEFT_EYE_PUPIL_INDEX = 385

        # Define the index for the bridge
        self._BRIDGR_R_INDEX = 193
        self._BRIDGR_C_INDEX = 168
        self._BRIDGR_L_INDEX = 417
        # Define the line type
        # PUPILS - a line connectong the corner of the eyes
        # HORIZONTAL - a line in the screen width going through the bridge point
        # BOTH - both lines
        self._LINE_TYPE = 'BOTH' # 'PUPILS' # 'HORIZONTAL'
        self.faces_landmarks = None
        self.pd_px = 0

    def process(self, frame):
        """
        Process the frame to detect the face and landmarks.

        Parameters:
        A single frame from a video or image source.

        Returns:
        None. The function updates the instance variables 'self.frame' and 'self.face_landmarks' with the processed frame and detected face landmarks, respectively.
        """
        if frame is None:
            return
        results = self.face_mesh.process(frame)
        if results.multi_face_landmarks and results.multi_face_landmarks[0]:
            self.frame = frame
            self.face_landmarks = results.multi_face_landmarks[0]

    def get_landmark_coordinates(self, landmark_index):
        """
        Get the coordinates (in the screen's coordinate system) of a specific landmark in the processed frame.

        Parameters:
        landmark_index (int): The index of the landmark in the face_landmarks.landmark list.

        Returns:
        Point: A Point object containing the x and y coordinates of the landmark in pixels.
               Returns None if the frame is not processed or the landmark index is invalid.
        """
        if self.frame is None:
            return None
        h, w, _ = self.frame.shape        
        landmark = self.face_landmarks.landmark[landmark_index]
        return utils.coordinates_to_pixles(w, h, landmark.x, landmark.y)

    def calculate_reflection_height_old(self, landmark_of_interest):
        if self.frame is None:
            return None
        h, w, _ = self.frame.shape
        midImageHeight = h/2
        reqiredHeight = int(midImageHeight) #set default to screen center
        point_of_interest_coordinates = self.get_landmark_coordinates(landmark_of_interest)
        if point_of_interest_coordinates is None:
            return reqiredHeight
        dy_pixels = midImageHeight-point_of_interest_coordinates.y
        dy_mm = dy_pixels * self.ccd_px_size * self.ccd_height_px / h

        alpha = np.arctan(dy_mm / self.focal_length)
        phi = self.ccd_ang - alpha
        reqiredHeight = self.patient_distance * np.tan(phi) - self.camera_above_screen
        reqiredHeight = int(reqiredHeight * self.screen_height_px / self.screen_height_mm)
        return reqiredHeight
    
    def calculate_reflection_height(self, landmark_of_interest):
        if self.frame is None:
            return None
        
        normal_to_ccd = np.array([np.tan(self.horizontal_camera_angle), np.tan(self.vertical_camera_angle), -1])
        normal_to_mirror = np.array([0, 0, -1])
        ccd_angle = np.arccos(np.dot(normal_to_ccd, normal_to_mirror) / np.linalg.norm(normal_to_ccd))
        
        u1 = np.linalg.norm(np.array([1, 0, np.tan(self.horizontal_camera_angle)]))
        u2 = np.linalg.norm(np.array([0, 1, np.tan(self.vertical_camera_angle)]))
        CCD_to_Mirror_Transformation_Matrix = np.array(u1, u2, normal_to_ccd)
        Mirror_to_CCD_Transformation_Matrix = np.linalg.inv(CCD_to_Mirror_Transformation_Matrix)
        v = np.array([0, self.screen_height_mm, 0])
        v_tilda = np.matmul(Mirror_to_CCD_Transformation_Matrix, v)
        d_tilda = np.linalg.norm(v_tilda)
        
        h, w, _ = self.frame.shape
        midImageWidth = w/2
        midImageHeight = h/2
        reqiredHeight = int(midImageHeight) #set default to screen center
        point_of_interest_coordinates = self.get_landmark_coordinates(landmark_of_interest)
        if point_of_interest_coordinates is None:
            return reqiredHeight
        d_pixels = np.sqrt((midImageWidth-point_of_interest_coordinates.x)**2 + (midImageHeight-point_of_interest_coordinates.y)**2)
        d_mm = d_pixels * self.ccd_px_size

        alpha = np.arctan(d_mm / self.focal_length)
        phi = self.ccd_ang + alpha
        patient_dist = self.calculate_patient_distance2()
        if patient_dist is None:
            return reqiredHeight
        y_tilda = patient_dist * np.tan(phi) + d_tilda

        u_tilda = np.array([0, y_tilda, 0])
        u = np.matmul(CCD_to_Mirror_Transformation_Matrix, u_tilda)
        reqiredHeight = np.linalg.norm(u)
        reqiredHeight = int(reqiredHeight * self.screen_height_px / self.screen_height_mm)
        return reqiredHeight
    
    def calculate_patient_height(self):
        return self.calculate_reflection_height(self._BRIDGR_C_INDEX)

    def calculate_patient_distance(self):
        if self.frame is None:
            return None
        h, w, _ = self.frame.shape
        midImageHeight = h/2
        patientDistance = 0
        bridge_center_height = self.calculate_reflection_height(self._BRIDGR_C_INDEX)            
        dy_pixels = midImageHeight-bridge_center_height
        dy_mm = dy_pixels * self.ccd_px_size * self.ccd_height_px / h
        pupil_r = self.get_landmark_coordinates(self._LEFT_EYE_PUPIL_INDEX)
        pupil_l = self.get_landmark_coordinates(self._RIGHT_EYE_PUPIL_INDEX)
        pd_px = int(np.sqrt(np.square(pupil_r.x-pupil_l.x)+np.square(pupil_r.y-pupil_l.y)))
        pd_mm = 60
        px2mm = pd_mm / pd_px
        alpha = np.arctan(dy_mm / self.focal_length)
        phi = self.ccd_ang - alpha
        a = dy_pixels * px2mm
        c = a / np.sin(alpha)
        patientDistance = c * np.cos(phi)
        return patientDistance
    
    def display_debug_frame(self):
        if self.frame is None:
            return
        h, w, _ = self.frame.shape
        left_eye = self.get_landmark_coordinates(self._LEFT_EYE_PUPIL_INDEX)
        right_eye = self.get_landmark_coordinates(self._RIGHT_EYE_PUPIL_INDEX)
        # Draw the pupils
        cv2.circle(self.frame, (left_eye.x, left_eye.y), 3, (255, 0, 0), -1)
        cv2.circle(self.frame, (right_eye.x, right_eye.y), 3, (255, 0, 0), -1)
        # Draw a line connecting the pupils
        cv2.line(self.frame, (left_eye.x, left_eye.y), (right_eye.x, right_eye.y), (0, 255, 0), 3)

        left_eye_reflection_height = self.calculate_reflection_height(self._LEFT_EYE_PUPIL_INDEX)
        right_eye_reflection_height = self.calculate_reflection_height(self._RIGHT_EYE_PUPIL_INDEX)
        # Draw the pupils
        cv2.circle(self.frame, (left_eye.x, left_eye_reflection_height), 3, (255, 0, 0), -1)
        cv2.circle(self.frame, (right_eye.x, right_eye_reflection_height), 3, (255, 0, 0), -1)
        # Draw a line connecting the pupils
        cv2.line(self.frame, (left_eye.x, left_eye_reflection_height), (right_eye.x, right_eye_reflection_height), (255, 0, 0), 3)

        reflection_height = el.calculate_reflection_height(self._BRIDGR_C_INDEX)
        cv2.line(self.frame, (0, reflection_height), (w, reflection_height), (0, 0, 255), 3)
        
if __name__ == "__main__":
    # Read the database directory from the first line of the config file
    with open('config.txt', 'r') as file:
        db_dir = file.readline().strip()  # Read the first line and remove any trailing whitespace
    db_path = os.path.join(db_dir,r'SparkSync.bytes')
    
    #create instance of the SparkEyeLevel class
    el = SparkEyeLevel(True)
    #create webcam capture manager
    webcam = WebcamCapture()
    #initilize DB monitor
    dm = DBMonitor(db_path)
    
    try:
        webcam.init()
        webcam.start()
        while True:
            _frame = webcam.get_frame()
            if _frame is not None:
                el.process(_frame)
                reflection_height = el.calculate_patient_height()
                patient_distance = el.calculate_patient_distance()
                print(f"Reflection height: {reflection_height} mm, Patient Distance: {patient_distance} mm")
                el.display_debug_frame()
                cv2.imshow("Pupil Connection", _frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break
    except Exception as e:
        print(f"failed to process Eye Level. Error: {e.args[0]}")

    finally:
        webcam.stop()
        webcam.release()
        cv2.destroyAllWindows()