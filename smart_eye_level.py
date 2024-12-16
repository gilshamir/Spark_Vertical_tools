import cv2
import mediapipe as mp
from DBMonitor import DBMonitor
import os

IS_DEBUG = True

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define indices for the pupil approximation
#https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png
RIGHT_EYE_NASAL_INDEX = 133
LEFT_EYE_NASAL_INDEX = 362
RIGHT_EYE_TEMPORAL_INDEX = 33
LEFT_EYE_TEMPORAL_INDEX = 263

# Define the index for the bridge
BRIDGR_R_INDEX = 193
BRIDGR_C_INDEX = 168
BRIDGR_L_INDEX = 417
# Define the line type
# PUPILS - a line connectong the corner of the eyes
# HORIZONTAL - a line in the screen width going through the bridge point
# BOTH - both lines
LINE_TYPE = 'BOTH' # 'PUPILS' # 'HORIZONTAL'

#set DB path
# Open the config file and read the first line
with open('config.txt', 'r') as file:
    db_dir = file.readline().strip()  # Read the first line and remove any trailing whitespace

db_path = os.path.join(db_dir,r'SparkSync.bytes')

# Open the webcam
cap = cv2.VideoCapture(1)

#initilize DB monitor
dm = DBMonitor(db_path)

def coordinates_to_pixles(w,h,x,y):
    return (int(x * w), int(y * h))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Flip the image horizontally for a selfie-view display
    frame = cv2.flip(frame, 1)
    # Convert the image to RGB (MediaPipe requires this)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame to detect the face and landmarks
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Convert normalized coordinates to pixel coordinates
            h, w, _ = frame.shape
            if LINE_TYPE == 'HORIZONTAL' or LINE_TYPE == 'BOTH':
                bridge_r = face_landmarks.landmark[BRIDGR_C_INDEX]
                bridge_c = face_landmarks.landmark[BRIDGR_C_INDEX]
                bridge_l = face_landmarks.landmark[BRIDGR_C_INDEX]
                (bridge_r_x_coord, bridge_r_y_coord) = coordinates_to_pixles(w,h,bridge_r.x,bridge_r.y)
                (bridge_c_x_coord, bridge_c_y_coord) = coordinates_to_pixles(w,h,bridge_c.x,bridge_c.y)
                (bridge_l_x_coord, bridge_l_y_coord) = coordinates_to_pixles(w,h,bridge_l.x,bridge_l.y)
                landmarks = [bridge_c_x_coord,bridge_c_y_coord]
                dm.set_FaceFeatures(landmarks)
                if IS_DEBUG:
                    cv2.line(frame, (0, bridge_c_y_coord), (w, bridge_c_y_coord), (0, 255, 255), 3)

            if LINE_TYPE == 'PUPILS' or LINE_TYPE == 'BOTH':
                # Get the pupil points by their indices
                right_eye = face_landmarks.landmark[LEFT_EYE_TEMPORAL_INDEX]
                left_eye = face_landmarks.landmark[RIGHT_EYE_TEMPORAL_INDEX]
                (right_eye_x_coord, right_eye_y_coord) = coordinates_to_pixles(w,h,right_eye.x,right_eye.y)
                (left_eye_x_coord, left_eye_y_coord) = coordinates_to_pixles(w,h,left_eye.x,left_eye.y)
                if IS_DEBUG:
                    # Draw the pupils
                    cv2.circle(frame, (left_eye_x_coord, left_eye_y_coord), 3, (255, 0, 0), -1)
                    cv2.circle(frame, (right_eye_x_coord, right_eye_y_coord), 3, (255, 0, 0), -1)
                    # Draw a line connecting the pupils
                    cv2.line(frame, (left_eye_x_coord, left_eye_y_coord), (right_eye_x_coord, right_eye_y_coord), (0, 255, 0), 3)
    if IS_DEBUG:
        # Display the frame
        cv2.imshow("Pupil Connection", frame)

    # Break the loop with the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# Release resources
cap.release()
cv2.destroyAllWindows()
