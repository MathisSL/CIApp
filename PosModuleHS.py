import cv2
import mediapipe as mp
import time
import math

class PoseDetector():

    def __init__(self, mode=False, upBody=False, smooth=True,
                 detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode, 
                                     model_complexity=1, 
                                     smooth_landmarks=self.smooth,
                                     min_detection_confidence=self.detectionCon, 
                                     min_tracking_confidence=self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, 
                                           self.mpPose.POSE_CONNECTIONS)
        return img
    
    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 4, (255, 0, 0), cv2.FILLED)
                    #cv2.putText(img, str(id), (cx + 8, cy + 8), cv2.FONT_HERSHEY_PLAIN,
                            #1, (0, 255, 0), 1)
        return self.lmList
    
    def findAngle(self, img, p1, p2, p3, draw=True):
        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]
        # Calculate the Angle
        angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2, x1-x2))
        if angle < 0:
            angle += 360
        # Choisir la couleur en fonction de l'angle
        goal = 180
        diff = abs(angle - goal)
        if diff <= 20:
            color = (0, 255, 0)  # Vert
        elif 20 < diff <= 30:
            color = (0, 165, 255)  # Orange
        else:
            color = (0, 0, 255)  # Rouge
        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), color, 2)
            cv2.line(img, (x2, y2), (x3, y3), color, 2)
            #cv2.line(img, (x1, y1), (x3, y3), color, 1)
            cv2.circle(img, (x1, y1), 6, color, cv2.FILLED)
            cv2.circle(img, (x2, y2), 6, color, cv2.FILLED)
            cv2.circle(img, (x3, y3), 6, color, cv2.FILLED)
            cv2.putText(img, str(int(angle)), (x2-120, y2+50), 
                        cv2.FONT_HERSHEY_PLAIN, 1.5, color, 2)
        return angle, diff, color

