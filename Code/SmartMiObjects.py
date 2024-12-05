import numpy as np
import cv2
import logging
import uuid
from scipy import linalg
from datetime import datetime
from enum import Enum


logging.basicConfig(filename='Score.log', level=logging.INFO, format='%(message)s',)
#logging.basicConfig(level=logging.INFO, format='%(message)s',)

class FaceFeatures:
    def __init__(self, faceImage_BGR, landmarks, imgShape):
        for l in landmarks.landmark:
            l.x = l.x * imgShape[1]
            l.y = l.y * imgShape[0]
        self.width = dist2D(landmarks.landmark[127], landmarks.landmark[356])
        self.height = dist2D(landmarks.landmark[10], landmarks.landmark[152])
        self.PD = dist2D(landmarks.landmark[468], landmarks.landmark[473])
        self.leftEyebrow, self.rightEyebrow = self.get_eyebrows(landmarks.landmark)
        self.leftUpperCheekLine, self.rightUpperCheekLine, self.leftLowerCheekLine, self.rightLowerCheekLine = self.get_cheek_lines(landmarks.landmark)
        self.jawLine = self.get_jaw_line(landmarks.landmark)
        self.noseWidth = dist2D(landmarks.landmark[193], landmarks.landmark[417])
        self.skinColor = self.calculate_skin_color(faceImage_BGR, landmarks.landmark)
        self.irisWidth = self.calculate_iris_width(landmarks.landmark)
        faceContour = self.jawLine.copy()
        S = [389, 251, 284, 332, 297, 338, 10, 109, 67, 103, 54, 21, 162]

        for i in S:
            faceContour = np.append(faceContour, [np.array([landmarks.landmark[i].x, landmarks.landmark[i].y]).astype(np.int32)], axis=0)
        self.faceArea = cv2.contourArea(faceContour)

        #self.eyeContoursCompletionLines = self.get_eye_contours_completion(landmarks.landmark)
        #self.upperNoseWidth = dist2D(landmarks.landmark[193], landmarks.landmark[417])
        #self.lowerNoseWidth = dist2D(landmarks.landmark[219], landmarks.landmark[278])

    def get_eyebrows(self, landmark):
        leftEyebrow = []
        rightEyebrow = []
        leftEyebrow.append(get_middle_point(landmark[336], landmark[285]))
        leftEyebrow.append(get_middle_point(landmark[296], landmark[295]))
        leftEyebrow.append(get_middle_point(landmark[334], landmark[282]))
        leftEyebrow.append(get_middle_point(landmark[293], landmark[283]))
        leftEyebrow.append(get_middle_point(landmark[300], landmark[276]))
        rightEyebrow.append(get_middle_point(landmark[107], landmark[55]))
        rightEyebrow.append(get_middle_point(landmark[66], landmark[65]))
        rightEyebrow.append(get_middle_point(landmark[105], landmark[52]))
        rightEyebrow.append(get_middle_point(landmark[63], landmark[53]))
        rightEyebrow.append(get_middle_point(landmark[70], landmark[46]))
        return leftEyebrow, rightEyebrow

    def get_cheek_lines(self, landmark):
        #leftUpperCheekLine = [[landmark[357].x, landmark[357].y], [landmark[350].x, landmark[350].y], [landmark[349].x, landmark[349].y], [landmark[348].x, landmark[348].y], 
        leftUpperCheekLine = [[landmark[349].x, landmark[349].y], [landmark[348].x, landmark[348].y], 
                        [landmark[347].x, landmark[347].y], [landmark[346].x, landmark[346].y], [landmark[340].x, landmark[340].y]]
        rightUpperCheekLine = [[landmark[111].x, landmark[111].y], [landmark[117].x, landmark[117].y], [landmark[118].x, landmark[118].y], [landmark[119].x, landmark[119].y], 
                        [landmark[120].x, landmark[120].y]]
                        #[landmark[121].x, landmark[121].y], [landmark[128].x, landmark[128].y]]
        
        p1 = get_middle_point(landmark[346], landmark[352])
        p2 = get_middle_point(landmark[347], landmark[280])
        p3 = get_middle_point(landmark[117], landmark[123])
        p4 = get_middle_point(landmark[118], landmark[50])
        leftLowerCheekLine = [p1, p2, [landmark[330].x, landmark[330].y]]
                                #[landmark[329].x, landmark[329].y], [landmark[277].x, landmark[277].y], [landmark[343].x, landmark[343].y]]
        rightLowerCheekLine = [p3, p4, [landmark[101].x, landmark[101].y]]
                                #[landmark[100].x, landmark[100].y], [landmark[47].x, landmark[47].y], [landmark[114].x, landmark[114].y]]

        leftUpperCheekLine = np.array(leftUpperCheekLine).astype(np.int32)
        rightUpperCheekLine = np.array(rightUpperCheekLine).astype(np.int32)
        leftLowerCheekLine = np.array(leftLowerCheekLine).astype(np.int32)
        rightLowerCheekLine = np.array(rightLowerCheekLine).astype(np.int32)
        return leftUpperCheekLine, rightUpperCheekLine, leftLowerCheekLine, rightLowerCheekLine

    def get_jaw_line(self, landmark):
        jawLine = [[landmark[127].x, landmark[127].y], [landmark[234].x, landmark[234].y], [landmark[93].x, landmark[93].y],
                    [landmark[58].x, landmark[58].y], [landmark[172].x, landmark[172].y], [landmark[136].x, landmark[136].y], [landmark[150].x, landmark[150].y],
                    [landmark[149].x, landmark[149].y], [landmark[176].x, landmark[176].y], [landmark[148].x, landmark[148].y], [landmark[152].x, landmark[152].y],
                    [landmark[377].x, landmark[377].y], [landmark[400].x, landmark[400].y], [landmark[378].x, landmark[378].y], [landmark[379].x, landmark[379].y],
                    [landmark[365].x, landmark[365].y], [landmark[397].x, landmark[397].y], [landmark[288].x, landmark[288].y], [landmark[435].x, landmark[435].y],
                    [landmark[401].x, landmark[401].y], [landmark[366].x, landmark[366].y], [landmark[447].x, landmark[447].y], [landmark[356].x, landmark[356].y]]

        jawLine = np.array(jawLine).astype(np.int32)
        return jawLine

    def calculate_skin_color(self, faceImage, landmark):
        landmarksList = [4, 205, 207, 425, 427, 108, 151, 337]
        colors = []
        for l in landmarksList:
            color = faceImage[int(landmark[l].y), int(landmark[l].x)]
            colors.append(color)

        skinColor = np.mean(np.array(colors), axis=0)
        return skinColor
    
    def calculate_iris_width(faceImage, landmark):
        leftIrisWidth = dist2D(landmark[476], landmark[474])
        rightIrisWidth = dist2D(landmark[469], landmark[471])
        irisWidth = (leftIrisWidth + rightIrisWidth) / 2
        return irisWidth

    def get_eye_contours_completion(self, landmark):
        p1 = get_middle_point(landmark[300], landmark[276])
        p2 = get_middle_point(landmark[336], landmark[285])
        p3 = get_middle_point(landmark[107], landmark[55])
        p4 = get_middle_point(landmark[70], landmark[46])
        p5 = get_middle_point(landmark[417], landmark[413])
        p6 = get_middle_point(landmark[193], landmark[189])

        eyeContoursCompletionLines = [np.array([p1, [landmark[383].x, landmark[383].y], [landmark[372].x, landmark[372].y], [landmark[340].x, landmark[340].y]]).astype(np.int32),
                                    np.array([p2, p5, [landmark[465].x, landmark[465].y], [landmark[357].x, landmark[357].y]]).astype(np.int32),
                                    np.array([p3, p6, [landmark[245].x, landmark[245].y], [landmark[128].x, landmark[128].y]]).astype(np.int32),
                                    np.array([p4, [landmark[156].x, landmark[156].y], [landmark[143].x, landmark[143].y], [landmark[111].x, landmark[111].y]]).astype(np.int32)]

        return eyeContoursCompletionLines

class Lenspair:
    def __init__(self, img_BGR, leftContour, rightContour):
        self.leftContour = leftContour
        self.rightContour = rightContour
        self.leftPupil = np.array([np.mean(leftContour[:,0]), np.mean(leftContour[:,1])])
        self.rightPupil = np.array([np.mean(rightContour[:,0]), np.mean(rightContour[:,1])])
        self.symmetryLine = np.array([[int((self.leftPupil[0] + self.rightPupil[0])/2)-1, int((self.leftPupil[1] + self.rightPupil[1])/2)+20], \
                                      [int((self.leftPupil[0] + self.rightPupil[0])/2), int((self.leftPupil[1] + self.rightPupil[1])/2)-20]])
        self.pointOfRotation = intersect_lines(self.symmetryLine, np.vstack((self.leftPupil, self.rightPupil)))
        self.transformedLeftContour = self.transform_contour(leftContour, self.pointOfRotation)
        self.transformedRightContour = self.transform_contour(rightContour, self.pointOfRotation)
        self.width = max(self.transformedLeftContour[:,0]) - min(self.transformedRightContour[:,0])
        self.leftHeight = max(self.transformedLeftContour[:,1]) - min(self.transformedLeftContour[:,1])
        self.leftWidth = max(self.transformedLeftContour[:,0]) - min(self.transformedLeftContour[:,0])
        self.rightHeight = max(self.transformedRightContour[:,1]) - min(self.transformedRightContour[:,1])
        self.rightWidth = max(self.transformedRightContour[:,0]) - min(self.transformedRightContour[:,0])
        self.DBL = min(self.transformedLeftContour[:,0]) - max(self.transformedRightContour[:,0])
        self.color = self.calculte_color(img_BGR)
        self.leftLensArea = cv2.contourArea(leftContour)
        self.rightLensArea = cv2.contourArea(rightContour)
        self.frameArea = self.leftLensArea + self.rightLensArea

    def transform_contour(self, contour, pointOfRotation):
        if self.symmetryLine[1,0] - self.symmetryLine[0,0] == 0:
            return contour
        
        symmetryLineAngle = np.arctan((self.symmetryLine[1,1] - self.symmetryLine[0,1]) / (self.symmetryLine[1,0] - self.symmetryLine[0,0]))
        if symmetryLineAngle < 0:
            rotationAngle = symmetryLineAngle + np.pi/2
        else:
            rotationAngle = symmetryLineAngle - np.pi/2

        rotationMatrix = np.array([[np.cos(rotationAngle), np.sin(rotationAngle), 0],
                                   [-np.sin(rotationAngle), np.cos(rotationAngle), 0],
                                   [0, 0, 1]])
        translationMatrix = [[1, 0, -pointOfRotation[0]],
                             [0, 1, -pointOfRotation[1]],
                             [0, 0, 1]]
        
        transformationMatrix = np.matmul(np.linalg.inv(translationMatrix), np.matmul(rotationMatrix, translationMatrix))
        contour = np.hstack((contour, np.ones((contour.shape[0],1))))
        transformedContour = np.matmul(transformationMatrix, np.transpose(contour))
        return np.transpose(transformedContour)[:,0:-1]

    def calculte_color(self, img):
        leftLensColor = self.calculte_contour_color(img, 'L')
        rightLensColor = self.calculte_contour_color(img, 'R')
        color = np.rint((leftLensColor + rightLensColor) / 2)       # Assuming BGR
        return color

    def calculte_contour_color(self, img, chiraliry):
        if chiraliry == 'L':
            pupil = self.leftPupil
            contour = self.leftContour
        else:
            pupil = self.rightPupil
            contour = self.rightContour

        d = 5
        phi = np.arctan((contour[:,0] - pupil[0]) / (contour[:,1] - pupil[1]))
        phi[contour[:,1] < pupil[1]] = phi[contour[:,1] < pupil[1]] + np.pi
        extendedContour = np.array(contour + d * np.column_stack((np.sin(phi), np.cos(phi)))).astype(np.int32)
        #for i in range(extendedContour.shape[0]):
        #    cv2.circle(img, (extendedContour[i,0], extendedContour[i,1]), 1, (0, 255, 0), -1)
        #half = cv2.resize(img, (0, 0), fx = 0.4, fy = 0.4)
        #cv2.imshow("", half)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        pixelColors = img[extendedContour[:,1], extendedContour[:,0]]
        color = np.mean(pixelColors, axis=0)
        return color

class Score:
    def __init__(self, faceFeatures, lenspair):
        self.faceFeatures = faceFeatures
        self.lenspair = lenspair
        self.frameWidth, self.frameWidthRatio = self.calculate_frame_width_score()
        self.eyebrowsMatch, self.leftEyebrow, self.rightEyebrow = self.calculate_eyebrows_match_score()
        self.lowerCheekLine, self.leftCheekLine, self.rightCheekLine = self.calculate_lower_cheek_line_score()
        self.frameShape, self.circularityPanelty, self.faceAspectRatio, self.faceCircularity, self.face_circleCenter, self.face_circleRadius, \
                self.leftLens_circleCenter, self.leftLens_circleRadius, self.rightLens_circleCenter, self.rightLens_circleRadius = self.frame_shape_score()
        self.DBL, self.DBLWidthRatio = self.calculate_DBL_score()
        self.frameColor, self.colorDiff = self.calculate_frame_color_score()
        self.frameArea, self.frameAreaRatio = self.calculate_frame_area_score()
        
        self.W = self.get_weights()
        self.total = self.get_total_score()

    def calculate_frame_width_score(self):
        ratio = self.lenspair.width / self.faceFeatures.width
        frameWidthScore = max(round(self.parabola_score(ratio, 1, -150), 2), 0)

        logging.info(f'Lenspair width: {self.lenspair.width}')
        logging.info(f'Face width: {self.faceFeatures.width}')
        logging.info(f'ratio: {ratio}')
        logging.info(f'frameWidthScore: {frameWidthScore}')
        logging.info(f'')
        return frameWidthScore, round(ratio, 2)
    
    def calculate_eyebrows_match_score(self):
        leftEyebrowScore = self.calculate_single_eyebrow_score('Left')
        rightEyebrowScore = self.calculate_single_eyebrow_score('Right')
        eyebrowsMatchScore = round((leftEyebrowScore + rightEyebrowScore) / 2, 2)

        logging.info(f'leftEyebrowScore: {leftEyebrowScore}')
        logging.info(f'rightEyebrowScore: {rightEyebrowScore}')
        logging.info(f'eyebrowsMatchScore: {eyebrowsMatchScore}')
        logging.info('')
        return eyebrowsMatchScore, round(leftEyebrowScore, 2), round(rightEyebrowScore, 2)
    
    def calculate_single_eyebrow_score(self, chirality):
        if chirality == 'Left':
            eyebrow = self.faceFeatures.leftEyebrow
            lensContour = self.lenspair.leftContour
        else:
            eyebrow = self.faceFeatures.rightEyebrow
            lensContour = self.lenspair.rightContour
        
        lensContourClosestPoints = self.get_lens_contour_closest_points(eyebrow, lensContour)

        diffs = []
        for i in range(len(lensContourClosestPoints)):
            d_scaled = np.sqrt(sum((lensContourClosestPoints[i] - eyebrow[i])**2)) / self.faceFeatures.irisWidth       # Scaled pixel distance
            diffs.append(d_scaled)

        diffs = np.array(diffs)
        mean = np.mean(diffs)
        std = np.std(diffs)
        eyebrowScore = self.double_parabola_score(mean, std, 0.1, 0, -10, -150, 0.6, 0.4)
        return eyebrowScore
    
    def calculate_lower_cheek_line_score(self):
        leftCheekLineScore = self.calculate_single_cheek_line_score('Left')
        rightCheekLineScore = self.calculate_single_cheek_line_score('Right')
        lowerCheekLineScore = round((leftCheekLineScore + rightCheekLineScore) / 2, 2)

        logging.info(f'leftCheekLineScore: {leftCheekLineScore}')
        logging.info(f'rightCheekLineScore: {rightCheekLineScore}')
        logging.info(f'lowerCheekLineScore: {lowerCheekLineScore}')
        logging.info('')
        return lowerCheekLineScore, round(leftCheekLineScore, 2), round(rightCheekLineScore, 2)
    
    def calculate_single_cheek_line_score(self, chirality):
        if chirality == 'Left':
            cheekLine = self.faceFeatures.leftLowerCheekLine
            lensContour = self.lenspair.leftContour
        else:
            cheekLine = self.faceFeatures.rightLowerCheekLine
            lensContour = self.lenspair.rightContour
        lensContourClosestPoints = self.get_lens_contour_closest_points(cheekLine, lensContour)

        cheekLineScore = 10
        for i in range(len(lensContourClosestPoints)):
            d = cheekLine[i,1] - lensContourClosestPoints[i,1]
            if d <= 0:
                cheekLineScore = 0
                break

        return cheekLineScore

    def frame_shape_score(self):
        face_circleCenter, face_circleRadius, _, faceCircularity = fit_circle(self.faceFeatures.jawLine)
        faceAspectRatio = self.faceFeatures.height / self.faceFeatures.width
        leftLens_circleCenter, leftLens_circleRadius, _, leftLens_circularity = fit_circle(self.lenspair.leftContour)
        #leftLensAspectRatio = self.lenspair.leftHeight / self.lenspair.leftWidth
        rightLens_circleCenter, rightLens_circleRadius, _, rightLens_circularity = fit_circle(self.lenspair.rightContour)
        #rightLensAspectRatio = self.lenspair.rightHeight / self.lenspair.rightWidth
        lenspair_circularity = (leftLens_circularity + rightLens_circularity) / 2
        #lenspairAspectRatio = (leftLensAspectRatio + rightLensAspectRatio) / 2

        aspectRatioUpperThreshold = 1.25
        aspectRatioLowerThreshold = 1
        ellipticalityUpperThreshold = 0.7
        ellipticalityLowerThreshold = 0.3

        circularityPanelty = 5 * (1 - abs(faceCircularity - lenspair_circularity))
        if faceAspectRatio <= aspectRatioLowerThreshold:
            aspectRatioWeight = 0
        else:
            m = 1 / (aspectRatioUpperThreshold - aspectRatioLowerThreshold)
            n = 0 - m * aspectRatioLowerThreshold
            aspectRatioWeight = min(m * faceAspectRatio + n, 1)
        if faceCircularity <= ellipticalityLowerThreshold:
            ellipticalityWeight = 0
        else:
            m = 1 / (ellipticalityUpperThreshold - ellipticalityLowerThreshold)
            n = 0 - m * ellipticalityLowerThreshold
            ellipticalityWeight = min(m * faceCircularity + n, 1)

        frameShapeScore = round(max(10 - circularityPanelty * (1 - aspectRatioWeight * ellipticalityWeight), 0), 2)

        logging.info(f'faceAspectRatio: {faceAspectRatio}')
        logging.info(f'faceCircularity: {faceCircularity}')
        logging.info(f'leftLens_circularity: {leftLens_circularity}')
        logging.info(f'rightLens_circularity: {rightLens_circularity}')
        logging.info(f'circularityPanelty: {circularityPanelty}')
        logging.info(f'aspectRatioWeight: {aspectRatioWeight}')
        logging.info(f'ellipticalityWeight: {ellipticalityWeight}')
        logging.info(f'frameShapeScore: {frameShapeScore}')
        logging.info('')
        return frameShapeScore, round(circularityPanelty, 2), round(faceAspectRatio, 2), round(faceCircularity, 2),\
                face_circleCenter, face_circleRadius, leftLens_circleCenter, leftLens_circleRadius, rightLens_circleCenter, rightLens_circleRadius
    
    def calculate_DBL_score(self):
        ratio = self.lenspair.DBL / self.faceFeatures.noseWidth
        if ratio <= 1.2:
            DBLScore = 10
        elif ratio >= 2:
            DBLScore = 0
        else:
            m = -10 / 0.8
            n = 10 - m * 1.2
            DBLScore = round(m * ratio + n, 2)

        logging.info(f'DBL: {self.lenspair.DBL}')
        logging.info(f'Face nose width: {self.faceFeatures.noseWidth}')
        logging.info(f'ratio: {ratio}')
        logging.info(f'DBL: {DBLScore}')
        logging.info(f'')
        return DBLScore, round(ratio, 2)
            
    def calculate_frame_color_score(self):
        diff = np.sqrt((self.lenspair.color[0] - self.faceFeatures.skinColor[0])**2 + 
                       (self.lenspair.color[1] - self.faceFeatures.skinColor[1])**2 + (self.lenspair.color[2] - self.faceFeatures.skinColor[2])**2)

        if diff >= 100:
            frameColorScore = 10
        else:
            m = 10/100
            n = 0
            frameColorScore = max(round(m * diff + n, 2), 0)

        logging.info(f'diff: {diff}')
        logging.info(f'frameColorScore: {frameColorScore}')
        logging.info(f'')
        return frameColorScore, round(diff, 2)
    
    def calculate_frame_area_score(self):
        ratio = self.lenspair.frameArea / self.faceFeatures.faceArea
        if ratio >= 0.16 and ratio <= 0.18:
            frameAreaScore = 10
        elif ratio < 0.16:
            m = 10/0.04
            n = 10 - m * 0.16
            frameAreaScore = max(round(m * ratio + n, 2), 0)
        else:
            m = -10/0.06
            n = 10 - m * 0.18
            frameAreaScore = max(round(m * ratio + n, 2), 0)

        logging.info(f'Lenspair area: {self.lenspair.frameArea}')
        logging.info(f'Face area: {self.faceFeatures.faceArea}')
        logging.info(f'ratio: {ratio}')
        logging.info(f'frameAreaScore: {frameAreaScore}')
        logging.info(f'')
        return frameAreaScore, round(ratio, 3)

    def get_weights(self):
        defaultWeights = {"frameWidth": 0.3, "eyebrowsMatch": 0.2, "lowerCheekLine": 0.05, "frameShape": 0.1, "DBL": 0.1, "frameColor": 0.1, "frameArea": 0.15}
        w_frameWidth_factored = defaultWeights["frameWidth"] * 10 / max(self.frameWidth, 2)
        w_eyebrowsMatch_factored = defaultWeights["eyebrowsMatch"] * 10 / max(self.eyebrowsMatch, 2)
        w_lowerCheekLine_factored = defaultWeights["lowerCheekLine"]
        w_frameShape_factored = defaultWeights["frameShape"] * 10 / max(self.frameShape, 2)
        w_DBL_factored = defaultWeights["DBL"]
        w_frameColor_factored = defaultWeights["frameColor"]
        w_frameArea_factored = defaultWeights["frameArea"] * 10 / max(self.frameArea, 2)

        factoredWeights = {"frameWidth": w_frameWidth_factored, "eyebrowsMatch": w_eyebrowsMatch_factored, "lowerCheekLine": w_lowerCheekLine_factored, \
                           "frameShape": w_frameShape_factored, "DBL": w_DBL_factored, "frameColor": w_frameColor_factored, "frameArea": w_frameArea_factored}
        weights = normalize(factoredWeights)
        return weights

    def get_lens_contour_closest_points(self, facePoints, lensContour):
        closest_points = []
        for i in range(len(facePoints)):
            minDistance = np.inf
            for j in range(len(lensContour)):
                d = np.sqrt(sum((lensContour[j] - facePoints[i])**2))
                if d < minDistance:
                    idx = j
                    minDistance = d

            closest_points.append(lensContour[idx])
            
        return np.array(closest_points)

    def parabola_score(self, x, desiredX, a):
        b = -2*desiredX*a
        c = 10 - a*desiredX**2 - b*desiredX
        score = max(a*x**2 + b*x + c, 0)
        return score

    def double_parabola_score(self, x, y, desiredX, desiredY, a1, a2, w1, w2):
        score1 = self.parabola_score(x, desiredX, a1)
        score2 = self.parabola_score(y, desiredY, a2)
        score = max(w1 * score1 + w2 * score2, 0)
        return score

    def get_total_score(self):
        totalScore = round(self.W["frameWidth"] * self.frameWidth + self.W["eyebrowsMatch"] * self.eyebrowsMatch + self.W["lowerCheekLine"] * self.lowerCheekLine + \
                           self.W["frameShape"] * self.frameShape + self.W["DBL"] * self.DBL + self.W["frameColor"] * self.frameColor + self.W["frameArea"] * self.frameArea, 2)

        logging.info(f'totalScore: {totalScore}')
        logging.info('########################################################')
        logging.info('')
        return totalScore

class Result:
    def __init__(self, image, annotatedImage, appScore, userScore):
        self.image = image
        self.annotatedImage = annotatedImage
        self.appScore = appScore
        self.userScore = userScore
        now = datetime.now()
        self.timeSignature = now.strftime("%d-%m-%Y_%H-%M-%S")
        self.guid = uuid.uuid4()

class State(Enum):
    SPLASH = 1
    APP = 2


### Functions ###
def get_middle_point(p1, p2):
    return [int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)]

def intersect_lines(l1, l2):
    p1 = l1[0,:]
    p2 = l1[1,:]
    p3 = l2[0,:]
    p4 = l2[1,:]

    if p2[0] == p1[0]:
        x = p1[0]
        m2 = (p4[1] - p3[1]) / (p4[0] - p3[0])
        n2 = p3[1] - m2 * p3[0]
        y = m2 * x + n2
    else:
        m1 = (p2[1] - p1[1]) / (p2[0] - p1[0])
        n1 = p1[1] - m1 * p1[0]
        if (p4[0] - p3[0]) == 0:
            x = p3[0]
        else:
            m2 = (p4[1] - p3[1]) / (p4[0] - p3[0])
            n2 = p3[1] - m2 * p3[0]
            x = (n2 - n1) / (m1 - m2)
        y = m1 * x + n1

    return np.array([x,y])

def fit_circle(points):
    x = points[:,0]
    y = points[:,1]
    
    m = 3
    M = np.zeros([len(x), m])
    M[:, 0] = 1
    M[:, 1] = -2 * x
    M[:, 2] = -2 * y
    beta = np.array(-(x**2 + y**2))

    M_inv = linalg.pinv(M)
    fit = np.matmul(M_inv, beta)

    x0 = fit[1]
    y0 = fit[2]
    R2 = (x0**2 + y0**2) - fit[0]
    R = np.sqrt(R2)
    center = np.array([x0, y0])
    
    errors = (np.sqrt((x-x0)**2 + (y-y0)**2) - R) ** 2
    RMSE = np.sqrt(np.mean(errors))
    sum_of_squared_residuals = sum(errors)
    max_sum_of_squared_residuals = len(errors) * max(errors)
    score = 1 - (sum_of_squared_residuals / max_sum_of_squared_residuals)
    return center, R, RMSE, score

def normalize(dict):
    S = sum(dict.values())

    # Normalize values using Min-Max scaling
    normalized_dict = {}
    for key, value in dict.items():
        normalized_value = value / S
        normalized_dict[key] = normalized_value

    return normalized_dict

def dist2D(u, v):
    return np.sqrt((u.x - v.x) ** 2 + (u.y - v.y) ** 2)
def dist3D(u, v):
    return np.sqrt((u.x - v.x) ** 2 + (u.y - v.y) ** 2 + (u.z - v.z) ** 2)
