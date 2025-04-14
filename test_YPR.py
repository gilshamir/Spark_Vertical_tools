import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh.
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open a connection to the webcam.
cap = cv2.VideoCapture(1)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty frame.")
        continue

    # Convert the image to RGB.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    # If face landmarks are detected.
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            ih, iw = image.shape[:2]
            # Select landmark indices that correspond to reliable points:
            # Nose tip: index 1, Chin: index 152, Left eye outer corner: index 263,
            # Right eye outer corner: index 33, Left mouth corner: index 287, Right mouth corner: index 57.
            image_points = np.array([
                (face_landmarks.landmark[1].x * iw, face_landmarks.landmark[1].y * ih),    # Nose tip
                (face_landmarks.landmark[152].x * iw, face_landmarks.landmark[152].y * ih),  # Chin
                (face_landmarks.landmark[263].x * iw, face_landmarks.landmark[263].y * ih),  # Left eye outer corner
                (face_landmarks.landmark[33].x * iw, face_landmarks.landmark[33].y * ih),    # Right eye outer corner
                (face_landmarks.landmark[287].x * iw, face_landmarks.landmark[287].y * ih),  # Left mouth corner
                (face_landmarks.landmark[57].x * iw, face_landmarks.landmark[57].y * ih)     # Right mouth corner
            ], dtype="double")

            # Define corresponding 3D model points of a generic face (in mm).
            model_points = np.array([
                (0.0, 0.0, 0.0),             # Nose tip
                (0.0, -63.6, -12.5),         # Chin
                (-43.3, 32.7, -26.0),        # Left eye outer corner
                (43.3, 32.7, -26.0),         # Right eye outer corner
                (-28.9, -28.9, -24.1),       # Left mouth corner
                (28.9, -28.9, -24.1)         # Right mouth corner
            ])

            # Approximate camera internals.
            focal_length = iw  # or use: focal_length = 1.2 * iw, if needed
            center = (iw / 2, ih / 2)
            camera_matrix = np.array([
                [3.20688713e+03, 0.00000000e+00, 2.70261737e+02],
                [0.00000000e+00, 3.24445028e+03, 2.44411237e+02],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
            ], dtype="double")

            # Assuming no lens distortion.
            dist_coeffs = np.zeros((4, 1))

            # Solve for the rotation and translation vectors using solvePnP.
            success_pnp, rotation_vector, translation_vector = cv2.solvePnP(
                model_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success_pnp:
                # Convert rotation vector to rotation matrix.
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

                # Compute Euler angles from the rotation matrix.
                sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
                singular = sy < 1e-6
                if not singular:
                    x_angle = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                    y_angle = np.arctan2(-rotation_matrix[2, 0], sy)
                    z_angle = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
                else:
                    x_angle = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
                    y_angle = np.arctan2(-rotation_matrix[2, 0], sy)
                    z_angle = 0

                # Convert radians to degrees.
                pitch = np.degrees(x_angle)
                yaw = np.degrees(y_angle)
                roll = np.degrees(z_angle)

                # Display the head pose angles.
                cv2.putText(image, f"Pitch: {pitch:.2f}", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cv2.putText(image, f"Yaw: {yaw:.2f}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cv2.putText(image, f"Roll: {roll:.2f}", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                # Optionally, draw the landmark points on the image.
                for point in image_points:
                    cv2.circle(image, (int(point[0]), int(point[1])), 3, (0, 0, 255), -1)

    cv2.imshow("Head Pose Estimation", image)
    if cv2.waitKey(5) & 0xFF == 27:  # press ESC to quit
        break

cap.release()
face_mesh.close()
cv2.destroyAllWindows()
