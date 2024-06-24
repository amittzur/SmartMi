import subprocess
import json
import configparser
import cv2
import numpy as np
import os

class LensContourFinder:
    def __init__(self, path):
        self.path = path
    
    def get_frame_contours(self, frame):
        # Convert the frame to bytes
        success = cv2.imwrite('face.jpg', frame)
        if not success:
            raise Exception("Could not save face image.")
        
        relative_path = os.path.join(self.path,'SparkSdkApi','SparkAI.API.exe')
        parameters = [os.path.join(self.path,'face.jpg')]  # List of parameters

        # Combine the executable and parameters into a single command
        command = [relative_path] + parameters

        # Run the command
        result = subprocess.run(command, capture_output=True, text=True, cwd=os.path.join(self.path,'SparkSdkApi'))
        if result.returncode != 0:
            raise Exception("Failed to extract frame contours.")

        leftContour, rightContour = self.parse_frame_contour(os.path.join(self.path,"face_contour_points.json"))
        return leftContour, rightContour

    def parse_frame_contour(self, jsonFile): 
        # Read the JSON file
        with open(jsonFile, 'r') as file:
            contours = json.load(file)
        
        # Create lists for RightContour and LeftContour
        leftContour = [[point["X"], point["Y"]] for point in contours["LeftContour"]]
        rightContour = [[point["X"], point["Y"]] for point in contours["RightContour"]]
        leftContour = np.array(leftContour)
        rightContour = np.array(rightContour)
        return leftContour, rightContour

