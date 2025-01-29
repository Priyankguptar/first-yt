import cv2
import mediapipe as mp
import math
import numpy as np
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_holistic = mp.solutions.holistic
POSTURE_THRESHOLD = 160 
EAR_THRESHOLD = 0.25     
def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180.0 else 360 - angle

def eye_aspect_ratio(eye_points):
    """Calculate eye aspect ratio (EAR) for blink detection"""
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    return (A + B) / (2.0 * C)

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)      
    pose_results = pose.process(frame_rgb)
    face_results = face_mesh.process(frame_rgb)
    holistic_results = holistic.process(frame_rgb)

  
    if pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark
        
        
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, 
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, 
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        
        angle = calculate_angle(left_hip, left_knee, left_ankle)
        posture = "STANDING" if angle > POSTURE_THRESHOLD else "SITTING"
        
        
        cv2.putText(frame, f"Posture: {posture}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            
            left_eye = np.array([(face_landmarks.landmark[i].x, 
                                face_landmarks.landmark[i].y) 
                               for i in [362, 385, 387, 263, 373, 380]])
            
            right_eye = np.array([(face_landmarks.landmark[i].x, 
                                 face_landmarks.landmark[i].y) 
                                for i in [33, 160, 158, 133, 153, 144]])
            
            ear_left = eye_aspect_ratio(left_eye)
            ear_right = eye_aspect_ratio(right_eye)
            ear_avg = (ear_left + ear_right) / 2.0
            
            
            if ear_avg < EAR_THRESHOLD:
                cv2.putText(frame, "BLINKING", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    
    if holistic_results.face_landmarks:
        face_landmarks = holistic_results.face_landmarks.landmark
       
        mouth_top = face_landmarks[13].y
        mouth_bottom = face_landmarks[14].y
        mouth_distance = abs(mouth_top - mouth_bottom)
        
        if mouth_distance > 0.05:  
            cv2.putText(frame, "SMILING", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if pose_results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_styles
            .get_default_pose_landmarks_style()
        )

    cv2.putText(frame, f"FPS: {int(cap.get(cv2.CAP_PROP_FPS))}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    
    cv2.imshow('Advanced Pose Estimation', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()