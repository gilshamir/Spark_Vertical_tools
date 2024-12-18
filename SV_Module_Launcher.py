import cv2
import mediapipe as mp
from DBMonitor import DBMonitor
import os
import time
from Webcam import WebcamCapture
from SparkEyeLevel import SparkEyeLevel
from SparkHeadRotation import SparkHeadRotation

class SparkVerticalStateMachine:
    def __init__(self, spark_eye_level, spark_head_rotation, webcam, dm):
        self.state = None  # Initial state
        self.spark_eye_level = spark_eye_level
        self.spark_head_rotation = spark_head_rotation
        self.webcam = webcam
        self.dm = dm
        self._IS_DEBUG = True

    def transition(self):
        """
        Transitions between states based on the input command.
        """
        if self.current_state() == 1:
            try:
                self.webcam.init()
                self.webcam.start()
                while self.current_state() == 1:
                    _frame = self.webcam.get_frame()
                    if _frame is not None:
                        processed_frame, landmarks = self.spark_eye_level.process(_frame)
                        self.dm.set_FaceFeatures(landmarks)
                        cv2.imshow("Spark Eye Level", processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                        break
            finally:
                self.webcam.stop()
                self.webcam.release()
                cv2.destroyAllWindows()
        elif self.current_state() == 2:
            try:
                self.webcam.init()
                self.webcam.start()
                while self.current_state() == 2:
                    _frame = self.webcam.get_frame()
                    if _frame is not None:
                        processed_frame, landmarks = self.spark_head_rotation.process(_frame)
                        #self.dm.set_FaceFeatures(landmarks)
                        cv2.imshow("Spark Head Rotation", processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                        break
            finally:
                self.webcam.stop()
                self.webcam.release()
                cv2.destroyAllWindows()
        else:
            print(f"Nothing to do for state: {self.current_state()}")

    def current_state(self):
        return self.state
    
    def set_state(self, state):
        self.state = state


def setMachineState(state):
    sm.set_state(state)


# Read the database directory from the first line of the config file
with open('config.txt', 'r') as file:
    db_dir = file.readline().strip()  # Read the first line and remove any trailing whitespace
db_path = os.path.join(db_dir,r'SparkSync.bytes')

#create instances of the modules
eye_level = SparkEyeLevel(True)
head_rotation = SparkHeadRotation(True)

#create webcam capture manager
webcam = WebcamCapture()

#initilize DB monitor
db_monitor = DBMonitor(db_path)

#initilize the state machine
sm = SparkVerticalStateMachine(eye_level, head_rotation, webcam, db_monitor)

#define the callback function for the DBMonitor - this sets the state that is read from the DB
db_monitor.run(setMachineState)

#hold the previous state so that the state machine transitions will occur only once
previous_state = None

while True:
    current_state = sm.current_state() #get the current state
    if current_state != previous_state: #if it has changed
        sm.transition() #transit
        previous_state = current_state #save the state