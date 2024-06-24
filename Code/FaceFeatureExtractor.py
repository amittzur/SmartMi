import subprocess
import json
import mediapipe as mp
import cv2
import numpy as np
import os

class FaceFeatureExtractor:
    def __init__(self):
        # Initialize MediaPipe Face Detection
        self.mp_face = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face.FaceDetection(min_detection_confidence=0.2)

        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True)
    
    def get_face_mesh(self, image):
        return self.face_mesh.process(image)
    
    def detect_face(self, image):
        return self.face_detection.process(image)