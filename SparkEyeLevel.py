import cv2
import mediapipe as mp
from DBMonitor import DBMonitor
import os
from Webcam import WebcamCapture
from utils import utils
import numpy as np

class SparkEyeLevel:
    def __init__(self, debug=True):
        self._IS_DEBUG = debug 

        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.focal_length = 8 #in mm
        self.ccd_px_size = 0.006 #in mm
        self.ccd_height_px = 1280 #
        self.screen_height_mm = 700
        self.camera_above_screen = 55
        self.ccd_ang = 26 * np.pi / 180
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
        self.landmarks = []
        self.pd_px = 0
        
    def process(self, frame):
        # Process the frame to detect the face and landmarks
        results = self.face_mesh.process(frame)
        h, w, _ = frame.shape
        new_frame = np.zeros((h, w, 3), dtype=np.uint8)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Convert normalized coordinates to pixel coordinates
                if self._LINE_TYPE == 'HORIZONTAL' or self._LINE_TYPE == 'BOTH':
                    bridge_r = face_landmarks.landmark[self._BRIDGR_R_INDEX]
                    bridge_c = face_landmarks.landmark[self._BRIDGR_C_INDEX]
                    bridge_l = face_landmarks.landmark[self._BRIDGR_L_INDEX]
                    (bridge_r_x_coord, bridge_r_y_coord) = utils.coordinates_to_pixles(w,h,bridge_r.x,bridge_r.y)
                    (bridge_c_x_coord, bridge_c_y_coord) = utils.coordinates_to_pixles(w,h,bridge_c.x,bridge_c.y)
                    (bridge_l_x_coord, bridge_l_y_coord) = utils.coordinates_to_pixles(w,h,bridge_l.x,bridge_l.y)
                    self.landmarks = [bridge_c_x_coord,bridge_c_y_coord]

                    #pupils location
                    pupil_r = face_landmarks.landmark[self._LEFT_EYE_PUPIL_INDEX]
                    pupil_l = face_landmarks.landmark[self._RIGHT_EYE_PUPIL_INDEX]
                    (pupil_r_x_coord, pupil_r_y_coord) = utils.coordinates_to_pixles(w,h,pupil_r.x,pupil_r.y)
                    (pupil_l_x_coord, pupil_l_y_coord) = utils.coordinates_to_pixles(w,h,pupil_l.x,pupil_l.y)
                    self.pd_px = int(np.sqrt(np.square(pupil_r_x_coord-pupil_l_x_coord)+np.square(pupil_r_y_coord-pupil_l_y_coord)))
                    
                    #self.dm.set_FaceFeatures(landmarks)
                    if self._IS_DEBUG:
                        cv2.line(new_frame, (0, bridge_c_y_coord), (w, bridge_c_y_coord), (0, 255, 255), 3)

                if self._LINE_TYPE == 'PUPILS' or self._LINE_TYPE == 'BOTH':
                    # Get the pupil points by their indices
                    right_eye = face_landmarks.landmark[self._LEFT_EYE_TEMPORAL_INDEX]
                    left_eye = face_landmarks.landmark[self._RIGHT_EYE_TEMPORAL_INDEX]
                    (right_eye_x_coord, right_eye_y_coord) = utils.coordinates_to_pixles(w,h,right_eye.x,right_eye.y)
                    (left_eye_x_coord, left_eye_y_coord) = utils.coordinates_to_pixles(w,h,left_eye.x,left_eye.y)
                    if self._IS_DEBUG:
                        # Draw the pupils
                        cv2.circle(new_frame, (left_eye_x_coord, left_eye_y_coord), 3, (255, 0, 0), -1)
                        cv2.circle(new_frame, (right_eye_x_coord, right_eye_y_coord), 3, (255, 0, 0), -1)
                        # Draw a line connecting the pupils
                        cv2.line(new_frame, (left_eye_x_coord, left_eye_y_coord), (right_eye_x_coord, right_eye_y_coord), (0, 255, 0), 3)
        
        return (new_frame, self.landmarks)
    
    def calculate_projection_height(self, frame):
        h, w, _ = frame.shape
        reqiredHeight = int(h/2)
        if self.landmarks:
            bridge_x, bridge_y = self.landmarks
            midImageHeight = h/2
            dy_pixels = midImageHeight-bridge_y
            dy_mm = dy_pixels * self.ccd_px_size * self.ccd_height_px / h

            alpha = np.arctan(dy_mm / self.focal_length)
            phi = self.ccd_ang - alpha
            reqiredHeight = self.patient_distance * np.tan(phi) - self.camera_above_screen
            reqiredHeight = int(reqiredHeight * self.screen_height_px / self.screen_height_mm)
            if self._IS_DEBUG:
                print(reqiredHeight)
                cv2.line(frame, (0, reqiredHeight), (w, reqiredHeight), (0, 0, 222), 3)
                #cv2.line(frame, (0, int(h/2)), (w, int(h/2)), (0, 0, 222), 3)
        return reqiredHeight
        #return int(h/2)
        
    
    def calculate_patient_distance(self, frame):
        h, w, _ = frame.shape
        patientDistance = 0
        if self.landmarks:
            bridge_x, bridge_y = self.landmarks
            midImageHeight = h/2
            dy_pixels = midImageHeight-bridge_y
            dy_mm = dy_pixels * self.ccd_px_size * self.ccd_height_px / h

            pd_px =  self.pd_px
            pd_mm = 60
            px2mm = pd_mm / pd_px
            alpha = np.arctan(dy_mm / self.focal_length)
            #beta = np.pi / 2 - self.ccd_ang
            phi = self.ccd_ang - alpha
            a = dy_pixels * px2mm
            c = a / np.sin(alpha)
            patientDistance = c * np.cos(phi)

            if self._IS_DEBUG:
                print(patientDistance)
        return patientDistance
    
    def reset(self):
        self.landmarks = []
        
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
                processed_frame, landmarks = el.process(_frame)
                if landmarks:
                    display_height = el.calculate_projection_height(processed_frame)
                    #patient_distance = el.calculate_patient_distance(processed_frame)
                    #dm.set_FaceFeatures(landmarks)
                    #dm.set_FaceDisplayHeight(display_height)
                cv2.imshow("Pupil Connection", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break
    except Exception as e:
        print(f"failed to process Eye Level. Error: {e.args[0]}")
    finally:
        webcam.stop()
        webcam.release()
        cv2.destroyAllWindows()