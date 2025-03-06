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
        self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.1, min_tracking_confidence=0.1)

        self.focal_length = 6 #in mm
        self.ccd_px_size = 0.006 #in mm
        self.ccd_height_px = 1080 #
        self.screen_height_mm = 700
        self.camera_above_screen = 55
        self.ccd_ang = 26 * np.pi / 180
        #self.patient_distance = 600
        self.screen_height_px = utils.get_screen_dimensions()[1]
        self.faces_landmarks = None
        self.pd_px = []
        self.pd_mm = 60

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

            pupil_r = self.get_landmark_coordinates(self._LEFT_EYE_PUPIL_INDEX)
            pupil_l = self.get_landmark_coordinates(self._RIGHT_EYE_PUPIL_INDEX)
            pd_px = np.sqrt(np.square(pupil_r.x-pupil_l.x)+np.square(pupil_r.y-pupil_l.y))
            self.pd_px.append(pd_px)
            # Keep only the last 30 counts
            if len(self.pd_px) > 30:
                self.pd_px.pop(0)
        else:
            self.frame = None
            self.face_landmarks = None
            

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

    def calculate_reflection_height(self, landmark_of_interest):
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
        patient_dist = self.calculate_patient_distance2()
        if patient_dist is None:
            return reqiredHeight
        reqiredHeight = patient_dist * np.tan(phi) - self.camera_above_screen
        reqiredHeight = int(reqiredHeight * self.screen_height_px / self.screen_height_mm)
        return reqiredHeight
    
    def calculate_patient_height(self):
        return self.calculate_reflection_height(self._BRIDGR_C_INDEX)

    def calculate_patient_distance(self):
        if self.frame is None:
            return None
        h, w, _ = self.frame.shape
        midImageHeight = h/2
        patientDistance = None
        point_of_interest_coordinates = self.get_landmark_coordinates(self._BRIDGR_C_INDEX)
        if point_of_interest_coordinates is None:
            return None
        dy_pixels = midImageHeight-point_of_interest_coordinates.y
        dy_mm = dy_pixels * self.ccd_px_size * self.ccd_height_px / h
        pd_px = np.mean(self.pd_px)
        pd_mm = self.pd_mm
        px2mm = pd_mm / pd_px
        alpha = np.arctan(dy_mm / self.focal_length)
        if alpha == 0:
            return None
        phi = self.ccd_ang - alpha
        a = dy_pixels * px2mm
        c = a / np.sin(alpha)
        patientDistance = c * np.cos(phi)
        return int(patientDistance)
    
    def calculate_patient_distance2(self):
        if self.frame is None:
            return None
        patientDistance = None
        pd_px = np.mean(self.pd_px)
        pd_mm = self.pd_mm
        patientDistance = pd_mm / (pd_px * self.ccd_px_size) * self.focal_length
        return int(patientDistance)
    
    def display_debug_frame(self):
        if self.frame is None:
            return
        h, w, _ = self.frame.shape
        is_show_image = False
        debug_frame = self.frame if is_show_image else np.zeros((h, w, 3), np.uint8)
        left_eye = self.get_landmark_coordinates(self._LEFT_EYE_PUPIL_INDEX)
        right_eye = self.get_landmark_coordinates(self._RIGHT_EYE_PUPIL_INDEX)
        # Draw the pupils
        cv2.circle(debug_frame, (left_eye.x, left_eye.y), 3, (255, 0, 0), -1)
        cv2.circle(debug_frame, (right_eye.x, right_eye.y), 3, (255, 0, 0), -1)
        # Draw a line connecting the pupils
        cv2.line(debug_frame, (left_eye.x, left_eye.y), (right_eye.x, right_eye.y), (0, 255, 0), 3)

        left_eye_reflection_height = self.calculate_reflection_height(self._LEFT_EYE_PUPIL_INDEX)
        right_eye_reflection_height = self.calculate_reflection_height(self._RIGHT_EYE_PUPIL_INDEX)
        # Draw the pupils
        cv2.circle(debug_frame, (left_eye.x, left_eye_reflection_height), 3, (255, 0, 0), -1)
        cv2.circle(debug_frame, (right_eye.x, right_eye_reflection_height), 3, (255, 0, 0), -1)
        # Draw a line connecting the pupils
        cv2.line(debug_frame, (left_eye.x, left_eye_reflection_height), (right_eye.x, right_eye_reflection_height), (255, 0, 0), 3)

        patient_height = el.calculate_patient_height()
        cv2.line(debug_frame, (0, patient_height), (w, patient_height), (0, 0, 255), 3)
        cv2.imshow("debug frame", debug_frame)

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
                #reflection_height = el.calculate_patient_height()
                patient_distance = el.calculate_patient_distance()
                patient_distance2 = el.calculate_patient_distance2()
                #print(f"Patient Distance: {patient_distance} mm, Patient Distance2: {patient_distance2} mm")
                #print(f"Reflection height: {reflection_height} mm, Patient Distance: {patient_distance} mm")
                el.display_debug_frame()
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break
    except Exception as e:
        print(f"failed to process Eye Level. Error: {e.args[0]}")

    finally:
        webcam.stop()
        webcam.release()
        cv2.destroyAllWindows()