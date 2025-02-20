import cv2
import mediapipe as mp
from DBMonitor import DBMonitor
import os
import time
from Webcam import WebcamCapture
from SparkEyeLevel import SparkEyeLevel
from SparkHeadRotation import SparkHeadRotation
from states import State
import numpy as np

class SparkVerticalStateMachine:
    def __init__(self, spark_eye_level, spark_head_rotation, webcam, dm):
        self.state = None  # Initial state
        
        #All the modules are local properties of this state machine
        self.spark_eye_level = spark_eye_level
        self.spark_head_rotation = spark_head_rotation
        self.webcam = webcam
        self.dm = dm
        self._IS_DEBUG = True

    def transition(self):
        """
        Transitions between states based on the input command.
        """
        if self.current_state() == State.MeasurementStart.value:
            print(f"Current State: {State.MeasurementStart}")
        elif self.current_state() == State.Welcome.value:
            print(f"Current State: {State.Welcome}")
        elif self.current_state() == State.Position.value:
            print(f"Current State: {State.Position}")
        
            while self.current_state() == State.Position.value:
                _frame = self.webcam.get_frame()
                if _frame is not None:
                    processed_frame, _ = self.spark_eye_level.process(_frame)
                    patient_distance = self.spark_eye_level.calculate_patient_distance(_frame)
                    if (patient_distance < 700 and patient_distance > 500):
                        self.dm.set_UpdateState(State.NaturalPosture.value)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                    cv2.destroyAllWindows()
                    break
        elif self.current_state() == State.NaturalPosture.value:
            print(f"Current State: {State.NaturalPosture}")
            while self.current_state() == State.NaturalPosture.value:
                _frame = self.webcam.get_frame()
                if _frame is not None:
                    processed_frame, head_rotations, _, _ = self.spark_head_rotation.process(_frame)
                    if head_rotations >= 2:
                        self.dm.set_UpdateState(State.Gaze.value)
                        cv2.destroyAllWindows()
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                    cv2.destroyAllWindows()
                    break
        elif self.current_state() == State.Gaze.value:
            print(f"Current State: {State.Gaze}")
            prev_yaw = np.Infinity
            prev_pitch = np.Infinity
            while self.current_state() == State.Gaze.value:
                time.sleep(1)
                _frame = self.webcam.get_frame()
                if _frame is not None:
                    processed_frame, head_rotation_count, yaw, pitch = self.spark_head_rotation.process(_frame)
                    delta_yaw = np.abs(yaw-prev_yaw)
                    delta_pitch = np.abs(pitch-prev_pitch)
                    if delta_yaw <= 1 and delta_pitch <= 1:
                        self.dm.set_UpdateState(State.CustomerReadyForCapture.value)
                        cv2.destroyAllWindows()
                    else:
                        prev_pitch = pitch
                        prev_yaw = yaw
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                    cv2.destroyAllWindows()
                    break
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

try:
    sm.webcam.init()
    sm.webcam.start()
    while True:
        current_state = sm.current_state() #get the current state
        if current_state != previous_state: #if it has changed
            sm.transition() #transit
            previous_state = current_state #save the state
finally:
    sm.webcam.stop()
    sm.webcam.release()
    cv2.destroyAllWindows()