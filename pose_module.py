import cv2
import math
import mediapipe as mp

class PoseDetector:
    def __init__(self):
        # Compatible module loading for all versions
        try:
            self.mp_pose = mp.solutions.pose
            self.mp_draw = mp.solutions.drawing_utils
        except AttributeError:
            import mediapipe.python.solutions.pose as mp_pose
            import mediapipe.python.solutions.drawing_utils as mp_draw
            self.mp_pose = mp_pose
            self.mp_draw = mp_draw

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def find_pose(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        if self.results.pose_landmarks and draw:
            self.mp_draw.draw_landmarks(
                img, 
                self.results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS
            )
        return img

    def find_angle(self, img, p1, p2, p3, draw=True):
        if not self.results.pose_landmarks:
            return 0
        
        h, w, _ = img.shape
        lm = self.results.pose_landmarks.landmark
        
        # Keypoints Coordinates
        x1, y1 = int(lm[p1].x * w), int(lm[p1].y * h)
        x2, y2 = int(lm[p2].x * w), int(lm[p2].y * h)
        x3, y3 = int(lm[p3].x * w), int(lm[p3].y * h)

        # Angle Calculation
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360
        if angle > 180:
            angle = 360 - angle

        # Draw Angle Visuals
        if draw:
            cv2.circle(img, (x2, y2), 8, (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(int(angle)), (x2 - 20, y2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return angle