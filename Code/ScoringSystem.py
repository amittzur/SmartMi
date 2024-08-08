import numpy as np
import cv2
from SmartMiObjects import FaceFeatures, Lenspair, Score
from LensContourFinder import LensContourFinder
from FaceFeatureExtractor import FaceFeatureExtractor


class ScoringSystem:
    def __init__(self, image, basePath):
        self.image = image
        self.basePath = basePath
        self.cf = LensContourFinder(basePath)
        self.ffe = FaceFeatureExtractor()

    def calculate_score(self):
        # Get frame contours
        leftContour, rightContour = self.cf.get_frame_contours(self.image)
        self.lenspair = Lenspair(self.image, leftContour, rightContour)

        # Process the frame with MediaPipe Face Detection
        rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        results_mesh = self.ffe.get_face_mesh(rgb_image)

        if results_mesh.multi_face_landmarks:
            self.face_landmarks = results_mesh.multi_face_landmarks[0]
            self.faceFeatures = FaceFeatures(self.image, self.face_landmarks, self.image.shape)

        #results_detection = self.ffe.detect_face(image)

        #x, y, w, h = detect_face_location(image)
        #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 4)
        #cv2.circle(image, (x, y), 10, (0, 255, 255), -1)
        
        self.score = Score(self.faceFeatures, self.lenspair)
        return self.score
    
    def get_annotated_image(self):
        annotatedImage = self.image.copy()
        frameColor = (255,255,255)
        lineWidth = 5

        #cv2.polylines(annotatedImage, [np.array(self.lenspair.transformedLeftContour).astype(np.int32)], False, frameColor, lineWidth)
        #cv2.polylines(annotatedImage, [np.array(self.lenspair.transformedRightContour).astype(np.int32)], False, frameColor, lineWidth)
        cv2.polylines(annotatedImage, [np.array(self.lenspair.leftContour).astype(np.int32)], False, frameColor, lineWidth)
        cv2.polylines(annotatedImage, [np.array(self.lenspair.rightContour).astype(np.int32)], False, frameColor, lineWidth)

        self.draw_landmark_index(annotatedImage, 0, lineWidth)
        #mp_drawing.draw_landmarks(
        #     imgDebugMode, face_landmarks,
        #     mp_face_mesh.FACEMESH_NOSE | mp_face_mesh.FACEMESH_LEFT_EYEBROW | mp_face_mesh.FACEMESH_RIGHT_EYEBROW,
        #     landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=0),
        #     connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1))
        return annotatedImage

    def draw_landmark_index(self, annotatedImage, plotCircles, lineWidth):
        #S = [127, 356, 10, 152, 193, 417, 4, 101, 36, 330, 266, 108, 151, 337]
        S = [127, 356, 10, 152, 193, 417]

        for i in S:
            x, y = int(self.face_landmarks.landmark[i].x), int(self.face_landmarks.landmark[i].y)
            cv2.putText(annotatedImage, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), lineWidth, cv2.LINE_AA)
            cv2.circle(annotatedImage, (x, y), 10, (0, 255, 0), -1)  # Draw a small circle at the landmark position

        cv2.polylines(annotatedImage, [np.array(self.faceFeatures.leftEyebrow)], False, (0,0,255), lineWidth)
        cv2.polylines(annotatedImage, [np.array(self.faceFeatures.rightEyebrow)], False, (0,0,255), lineWidth)
        cv2.polylines(annotatedImage, [np.array(self.faceFeatures.leftUpperCheekLine)], False, (255,255,0), lineWidth)
        cv2.polylines(annotatedImage, [np.array(self.faceFeatures.rightUpperCheekLine)], False, (255,255,0), lineWidth)
        cv2.polylines(annotatedImage, [np.array(self.faceFeatures.leftLowerCheekLine)], False, (255,255,0), lineWidth)
        cv2.polylines(annotatedImage, [np.array(self.faceFeatures.rightLowerCheekLine)], False, (255,255,0), lineWidth)
        cv2.polylines(annotatedImage, [np.array(self.faceFeatures.jawLine)], False, (255,0,255), lineWidth)

        if plotCircles:
            cv2.circle(annotatedImage, tuple(self.score.face_circleCenter.astype(np.int32)), int(self.score.face_circleRadius), (76, 230, 150), lineWidth)
            cv2.circle(annotatedImage, tuple(self.score.leftLens_circleCenter.astype(np.int32)), int(self.score.leftLens_circleRadius), (76, 230, 150), lineWidth)
            cv2.circle(annotatedImage, tuple(self.score.rightLens_circleCenter.astype(np.int32)), int(self.score.rightLens_circleRadius), (76, 230, 150), lineWidth)

        return annotatedImage


