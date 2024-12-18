import cv2
import threading
from utils import utils

class WebcamCapture:
    def __init__(self, source=0):
        self.source = source
        self.capture = None
        self.frame = None
        self.running = False
        self.thread = None
        self.CAMERA_RESOLUTION = (1280, 720)
        self.SCREEN_RESOLUTION = utils.get_screen_dimensions()

    def _capture_loop(self):
        while self.running:
            ret, frame = self.capture.read()
            # Flip the image horizontally for a selfie-view display
            frame = cv2.flip(frame, 1)
            # Convert the image to RGB (MediaPipe requires this)
            #rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if ret:
                resized_frame = cv2.resize(frame, (self.SCREEN_RESOLUTION[0], self.SCREEN_RESOLUTION[1]))
                self.frame = resized_frame  # Update the latest frame

    def get_frame(self):
        if self.running:
            return self.frame
        return None
    
    def init(self):
        self.capture = cv2.VideoCapture(self.source)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.CAMERA_RESOLUTION[0])
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAMERA_RESOLUTION[1])

    def start(self):        
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread != None and self.thread.is_alive():
            self.thread.join()
    
    def release(self):
        if self.capture != None:
            self.capture.release()

# Usage Example
def main():
    webcam = WebcamCapture()

    try:
        webcam.init()
        webcam.start()
        while True:
            frame = webcam.get_frame()
            if frame is not None:
                cv2.imshow("Webcam Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break
    finally:
        webcam.stop()
        webcam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()