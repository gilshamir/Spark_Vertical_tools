import cv2
import mediapipe as mp
from DBMonitor import DBMonitor
import os
import time
from Webcam import WebcamCapture
from SparkEyeLevel import SparkEyeLevel
from SparkHeadRotation import SparkHeadRotation
from states import State, SubState
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
        if self.current_state() == State.CommercialVideoState.value:
            print(f"Current State: {State.CommercialVideoState}")
        elif self.current_state() == State.MeasurementStart.value:
            print(f"Current State: {State.MeasurementStart}")
        elif self.current_state() == State.Welcome.value:
            print(f"Current State: {State.Welcome}")
            self.dm.set_UpdateSubState(SubState.Start_Positioning.value)
        elif self.current_state() == State.Position.value:
            print(f"Current State: {State.Position}")
            self.dm.set_UpdateSubState(SubState.Start_Positioning.value)
            self.spark_head_rotation.reset()
            self.spark_head_rotation.resetBasePosture()
            too_close_index = 0
            too_far_index = 0
            correct_distance_index = 0
            user_identification_failed_index = 0
            repositioning_counter = 0
            set_initial_height_index = 0
            is_initial_height = True
            mean_patient_height = 0

            while self.current_state() == State.Position.value:
                _frame = self.webcam.get_frame()
                if _frame is not None:
                    self.spark_eye_level.process(_frame)
                    patient_distance = self.spark_eye_level.calculate_patient_distance()
                    patient_height = self.spark_eye_level.calculate_patient_height()
                    #print(f"patient_distance: {patient_distance}")
                    if (is_initial_height and patient_distance != None and patient_height != None):
                        if set_initial_height_index < 50:
                            set_initial_height_index = set_initial_height_index + 1
                            mean_patient_height = mean_patient_height*0.4 + patient_height*0.6
                        else:
                            is_initial_height = False
                            set_initial_height_index = 0
                            #print(f"frame shape: {_frame.shape}")
                            self.dm.set_FaceDisplayHeight(mean_patient_height)
                        #print(f"setting initial patient height: {patient_height}")
                    if (patient_distance == None):
                        user_identification_failed_index = user_identification_failed_index+1
                        if (user_identification_failed_index >= 40 and user_identification_failed_index <= 50):
                            self.dm.set_UpdateSubState(SubState.Start_Positioning.value)
                        if user_identification_failed_index >= 500:
                            user_identification_failed_index = 0
                            is_initial_height = True
                            #print("resseting patient height")
                            repositioning_counter = repositioning_counter+1
                            if repositioning_counter >= 3:
                                repositioning_counter = 0
                                self.dm.set_UpdateState(State.CommercialVideoState.value)
                            else:
                                self.dm.set_UpdateSubState(SubState.RePositioning.value)
                    elif (patient_distance!= None and patient_distance < 500):
                        too_close_index = too_close_index+1
                        correct_distance_index = 0
                        too_far_index = 0
                        user_identification_failed_index = 0
                        repositioning_counter = 0
                        if (too_close_index >= 40 and too_close_index <= 50):
                            self.dm.set_UpdateSubState(SubState.Start_Positioning.value)
                        if (too_close_index >= 350):
                            too_close_index = 0                            
                            self.dm.set_UpdateSubState(SubState.Backwards.value)
                    elif (patient_distance!= None and patient_distance > 650):
                        too_far_index = too_far_index+1
                        correct_distance_index = 0
                        too_close_index = 0
                        user_identification_failed_index = 0
                        repositioning_counter = 0
                        if (too_far_index >= 40 and too_far_index <= 50):
                            self.dm.set_UpdateSubState(SubState.Start_Positioning.value)
                        if (too_far_index >= 350):
                            too_far_index = 0
                            self.dm.set_UpdateSubState(SubState.Forward.value)
                    elif (patient_distance!= None and patient_distance < 650 and patient_distance > 500):
                        correct_distance_index = correct_distance_index + 1
                        repositioning_counter = 0
                        if (correct_distance_index >= 120):
                            correct_distance_index = 0
                            patient_height = self.spark_eye_level.calculate_patient_height()
                            self.dm.set_FaceDisplayHeight( patient_height)
                            #print(f"patient_distance: {patient_distance}")
                            #print(f"patient_height: {patient_height}")
                            self.dm.set_UpdateSubState(SubState.Done_Positioning.value)
                            time.sleep(0.9)                            
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                    cv2.destroyAllWindows()
                    break
        elif self.current_state() == State.NaturalPosture.value:
            print(f"Current State: {State.NaturalPosture}")
            self.dm.set_UpdateSubState(SubState.Start_head_Rotation.value)
            prev_yaw = np.Infinity
            prev_pitch = np.Infinity
            while self.current_state() == State.NaturalPosture.value:
                _frame = self.webcam.get_frame()
                if _frame is not None:
                    processed_frame, head_rotation_count, yaw, pitch = self.spark_head_rotation.process(_frame)
                    if yaw == None or pitch == None:
                        continue
                    if (prev_yaw != np.Infinity or prev_pitch != np.Infinity):
                        delta_yaw = np.abs(yaw-prev_yaw)
                        delta_pitch = np.abs(pitch-prev_pitch)
                        #print(f"yaw: {yaw}, pitch: {pitch}, d_yaw: {delta_yaw}, d_pitch: {delta_pitch}")
                        if delta_yaw >= 30 or delta_pitch >= 30:
                            self.dm.set_UpdateSubState(SubState.Head_In_Motion.value)
                            #print("moving head")
                        else:
                            self.dm.set_UpdateSubState(SubState.Head_Is_Static.value)
                            #print("static head")
                    prev_pitch = pitch
                    prev_yaw = yaw
                    time.sleep(0.2)
                    
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                    cv2.destroyAllWindows()
                    break
        elif self.current_state() == State.Gaze.value:
            print(f"Current State: {State.Gaze}")
            prev_yaw = np.Infinity
            prev_pitch = np.Infinity
            stability_index = 0
            while self.current_state() == State.Gaze.value:
                _frame = self.webcam.get_frame()
                if _frame is not None:
                    processed_frame, head_rotation_count, yaw, pitch = self.spark_head_rotation.process(_frame)
                    if yaw == None or pitch == None:
                        continue
                    delta_yaw = np.abs(yaw-prev_yaw)
                    delta_pitch = np.abs(pitch-prev_pitch)
                    if delta_yaw <= 1 and delta_pitch <= 1:
                        patient_height = self.spark_eye_level.calculate_patient_height()
                        #add average value
                        #self.dm.set_FaceDisplayHeight(patient_height)
                        self.dm.set_UpdateState(State.CustomerReadyForCapture.value)
                        cv2.destroyAllWindows()
                    else:
                        prev_pitch = pitch
                        prev_yaw = yaw
                        time.sleep(0.3)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                    cv2.destroyAllWindows()
                    break
        elif self.current_state() == State.CustomerReadyForCapture.value:
            print(f"Current State: {State.CustomerReadyForCapture}")
            #self.spark_eye_level.reset()
            #self.spark_head_rotation.reset()
        elif self.current_state() == State.CaptureStarted.value:
            print(f"Current State: {State.CaptureStarted}")
        elif self.current_state() == State.CaptureCompleted.value:
            print(f"Current State: {State.CaptureCompleted}")
        elif self.current_state() == State.CameraInHomePosition.value:
            print(f"Current State: {State.CameraInHomePosition}")
            #time.sleep(40)
            #self.dm.set_UpdateState(State.SparkResultsReady.value)
        elif self.current_state() == State.SparkResultsReady.value:
            print(f"Current State: {State.SparkResultsReady}")
            #time.sleep(1)
            #self.dm.set_UpdateState(State.MeasurementCompleted.value)
        elif self.current_state() == State.MeasurementCompleted.value:
            print(f"Current State: {State.MeasurementCompleted}")
            #time.sleep(1)
            #self.dm.set_UpdateState(State.CommercialVideoState.value)
        elif self.current_state() == State.GeneralMeasurementFault.value:
            print(f"Current State: {State.GeneralMeasurementFault}")
        elif self.current_state() == State.RetakePicture.value:
            print(f"Current State: {State.RetakePicture}")
        elif self.current_state() == State.SkipToNext.value:
            print(f"Current State: {State.SkipToNext}")
        elif self.current_state() == State.Idle.value:
            print(f"Current State: {State.Idle}")        
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
eye_level = SparkEyeLevel(False)
head_rotation = SparkHeadRotation(False)

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