from enum import Enum

class State(Enum):
    CommercialVideoState = 0
    MeasurementStart = 1
    Welcome = 2
    Position = 3
    NaturalPosture = 4
    Gaze = 5
    CustomerReadyForCapture = 6
    CaptureStarted = 7
    CaptureCompleted = 8
    CameraInHomePosition = 9
    SparkResultsReady = 10
    MeasurementCompleted = 11
    GeneralMeasurementFault = 12
    RetakePicture = 13
    SkipToNext = 14
    Idle = 15

class SubState(Enum):
    Start_Positioning = 0
    RePositioning = 1
    Forward = 2
    Backwards = 3
    Left = 4
    Right = 5
    Done_Positioning = 6
    Start_head_Rotation = 7
    Head_In_Motion = 8
    Head_Is_Static = 9
    Done_head_Rotation = 10
    