import cv2

class utils:

        @staticmethod
        def coordinates_to_pixles(w,h,x,y):
                return (int(x * w), int(y * h))
        
        @staticmethod
        def get_screen_dimensions():
                # Create a fullscreen window
                cv2.namedWindow("Screen", cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty("Screen", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                
                # Get the screen resolution
                screen_rect = cv2.getWindowImageRect("Screen")
                screen_width, screen_height = screen_rect[2], screen_rect[3]
                
                # Close the window
                cv2.destroyAllWindows()
                
                return screen_width, screen_height