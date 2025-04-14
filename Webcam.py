import cv2
import threading
from utils import utils

class WebcamCapture:
    def __init__(self, source=1):
        self.source = source
        self.capture = None
        self.frame = None
        self.running = False
        self.thread = None
        self.CAMERA_RESOLUTION = (1920, 1080)
        self.SCREEN_RESOLUTION = utils.get_screen_dimensions()

    def _capture_loop(self):
        while self.running:
            ret, frame = self.capture.read()
            # Flip the image horizontally for a selfie-view display
            frame = cv2.flip(frame, 1)
            # Convert the image to RGB (MediaPipe requires this)
            #rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if ret:
                #rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                cropped_image = self.crop_and_resize(frame, self.SCREEN_RESOLUTION[0], self.SCREEN_RESOLUTION[1]) #frame[:, 437:842]
                #resized_frame = cv2.resize(cropped_image, (self.SCREEN_RESOLUTION[0], self.SCREEN_RESOLUTION[1]))
                self.frame = cropped_image  # Update the latest frame
                #self.frame = frame
    def crop_and_resize(self, image, screen_width, screen_height):
        # Get screen aspect ratio
        screen_ar = screen_width / screen_height
        #print(f"Screen Aspect Ratio: {screen_ar}, Screen Width: {screen_width}, Screen Height: {screen_height}")

        # Get image dimensions
        img_height, img_width = image.shape[:2]
        img_ar = img_width / img_height
        #print(f"Image Aspect Ratio: {img_ar}, Image Width: {img_width}, Image Height: {img_height}")

        # Crop width if necessary
        if img_ar > screen_ar:
            new_width = int(screen_ar * img_height)
            left = (img_width - new_width) // 2
            right = left + new_width
            image = image[:, left:right]  # Crop width
            #print(f"image size after crop width: {image.shape}")

        # Resize using cv2
        image = cv2.resize(image, (screen_width, screen_height), interpolation=cv2.INTER_LANCZOS4)

        return image  # Return the processed image as a NumPy array
    
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
    except:
        print("Failed to start Webcam")
    finally:
        webcam.stop()
        webcam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()