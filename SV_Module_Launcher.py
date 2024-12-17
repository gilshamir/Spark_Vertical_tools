import cv2
import mediapipe as mp
from DBMonitor import DBMonitor
import os
import time
from Webcam import WebcamCapture
from SparkEyeLevel import SparkEyeLevel

class SparkVerticalStateMachine:
    def __init__(self, el, webcam, dm):
        self.state = None  # Initial state
        self.el = el
        self.webcam = webcam
        self.dm = dm

    def transition(self):
        """
        Transitions between states based on the input command.
        """
        if self.current_state() == 1:
            try:
                webcam.init()
                webcam.start()
                while self.current_state() == 1:
                    _frame = webcam.get_frame()
                    if _frame is not None:
                        processed_frame, landmarks = el.process(_frame)
                        dm.set_FaceFeatures(landmarks)
                        cv2.imshow("Pupil Connection", processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                        break
            finally:
                webcam.stop()
                webcam.release()
                cv2.destroyAllWindows()
        elif self.current_state() == 2:
            print(f"Doing stuff in status {self.current_state()}")
        else:
            print(f"Nothing to do for state: {self.current_state()}")

    def current_state(self):
        return self.state
    
    def set_state(self, state):
        self.state = state


def setMachineState(state):
    sm.set_state(state)



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
    sm = SparkVerticalStateMachine(el,webcam,dm)

    dm.run(setMachineState)
    
    previous_state = None

    while True:
        current_state = sm.current_state()
        if current_state != previous_state:
            sm.transition()
            previous_state = current_state