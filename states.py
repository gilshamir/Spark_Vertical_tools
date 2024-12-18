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