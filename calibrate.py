import cv2
import numpy as np

CAMERA_RESOLUTION = (1920, 1080)

# Define chessboard size
CHESSBOARD_SIZE = (6, 8)  # Number of internal corners in the board
SQUARE_SIZE = 0.010  # Real-world size of a square (meters, centimeters, etc.)

# Prepare object points (3D coordinates of chessboard corners)
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE

# Arrays to store object points and image points
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

# Capture frames from webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])

while len(imgpoints) < 25:  # Collect at least 10 samples
    ret, frame = cap.read()
    if not ret:
        break
    if cv2.waitKey(1) & 0xFF == ord('c'):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

        if found:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display corners
            cv2.drawChessboardCorners(frame, CHESSBOARD_SIZE, corners, found)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.imshow('Calibration', frame)

#cap.release()
#cv2.destroyAllWindows()

# Perform camera calibration
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Print results
print("Camera Matrix (Intrinsic Parameters):\n", camera_matrix)
print("Distortion Coefficients:\n", dist_coeffs)

# while True:  # Collect at least 10 samples
#     ret, frame = cap.read()
#     if not ret:
#         break
#     undistorted_img = cv2.undistort(frame, camera_matrix, dist_coeffs)
#     cv2.imshow('Calibration', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
cap.release()
cv2.destroyAllWindows()