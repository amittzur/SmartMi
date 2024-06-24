import numpy as np
import cv2
import PySimpleGUI as sg
import os.path
import pathlib
import sys
import mediapipe as mp
from win32api import GetSystemMetrics
from SmartMiObjects import FaceFeatures, Lenspair, Score

### Description ###
# SmartMi enables to display multiple images taken from a video stream simultaneously, in order to compare shown features.


### Functions ###
def get_full_path(file):
    if hasattr(sys, 'frozen'):
        basis = sys.executable
        basis = os.path.split(basis)[0]
    else:
        basis = pathlib.Path(__file__).parent.resolve().__str__()

    return os.path.join(basis, file)

def draw_landmark_index(image, landmarks, faceFeatures, score):
    #S = [127, 356, 10, 152, 468, 473, 193, 417, 58, 288]
    S = [127, 356, 10, 152]
    for i in S:
        #x, y = int(landmarks.landmark[i].x * image.shape[1]), int(landmarks.landmark[i].y * image.shape[0])
        x, y = int(landmarks.landmark[i].x), int(landmarks.landmark[i].y)
        cv2.putText(image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)  # Draw a small circle at the landmark position

    cv2.polylines(image, [np.array(faceFeatures.leftEyebrow)], False, (0,0,255), 1)
    cv2.polylines(image, [np.array(faceFeatures.rightEyebrow)], False, (0,0,255), 1)
    cv2.polylines(image, [np.array(faceFeatures.leftUpperCheekLine)], False, (255,255,0), 1)
    cv2.polylines(image, [np.array(faceFeatures.rightUpperCheekLine)], False, (255,255,0), 1)
    cv2.polylines(image, [np.array(faceFeatures.leftLowerCheekLine)], False, (255,255,0), 1)
    cv2.polylines(image, [np.array(faceFeatures.rightLowerCheekLine)], False, (255,255,0), 1)
    cv2.polylines(image, [np.array(faceFeatures.jawLine)], False, (255,0,255), 1)
    #cv2.polylines(image, faceFeatures.eyeContoursCompletionLines, False, (0,255,255), 1)
    #cv2.polylines(image, [nose], False, (0,0,255), 1)

    if type(score) != int:
        cv2.circle(image, tuple(score.face_circleCenter.astype(np.int32)), int(score.face_circleRadius), (76, 230, 150), 1)
        cv2.circle(image, tuple(score.leftLens_circleCenter.astype(np.int32)), int(score.leftLens_circleRadius), (76, 230, 150), 1)
        cv2.circle(image, tuple(score.rightLens_circleCenter.astype(np.int32)), int(score.rightLens_circleRadius), (76, 230, 150), 1)

def edit_score_text(score):
    scoreText = '''\
    Frame width ratio = {fw0}
    Frame width score = {fw}

    Left eyebrow score = {em0}
    Right eyebrow score = {em1}
    Eyebrows match score = {em}

    Left cheek line score = {lcl0}
    Right cheek line score = {lcl1}
    Lower cheek line score = {lcl}

    Circularity panelty = {fs0}
    Face aspect ratio = {fs1}
    Face circularity score = {fs2}
    Frame shape score = {fs}

    Total score = {t}\
    '''.format(fw0=score.frameWidthRatio, fw=score.frameWidthScore, em0=score.leftEyebrowScore, em1=score.rightEyebrowScore, em=score.eyebrowsMatchScore,\
               lcl0=score.leftCheekLineScore, lcl1=score.rightCheekLineScore, lcl=score.lowerCheekLineScore,\
                fs0=score.circularityPanelty, fs1=score.faceAspectRatio, fs2=score.face_circularityScore, fs=score.frameShapeScore, t=score.totalScore)

    scoreText = edit_rows(scoreText)
    return scoreText

def edit_rows(text):
    # Split the text into lines
    lines = text.split('\n')

    lines[1] = f"**{lines[1]}**"
    lines[5] = f"**{lines[5]}**"
    lines[9] = f"**{lines[9]}**"
    lines[14] = f"**{lines[14]}**"
    lines[16] = f"**{lines[16]}**"

    # Join the lines back together
    edited_text = '\n'.join(lines)

    return edited_text

def extract_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3, minSize=(30, 30))

    print("Found {0} faces!".format(len(faces)))
    if not len(faces):
        return []

    #for (x, y, w, h) in faces:
    #    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    (x, y, w, h) = faces[0]
    d = max(w, h)
    minX = max(0, x-int(d/2))
    maxX = min(frame.shape[1], x+int(3*d/2))
    minY = max(0, y-int(d/2))
    maxY = min(frame.shape[0], y+int(3*d/2))
    #frame = cv2.circle(frame, (x+int(d/2),y+int(d/2)), 5, (0, 0, 255)) 
    croppedFrame = frame[minY:maxY, minX:maxX]

    return croppedFrame

def set_images_for_display():
    global Images4Display

    if NumberOfImages == 0:
        Images4Display[:,:,0] = 0
        Images4Display[:,:,1] = 0
        Images4Display[:,:,2] = 193

    #imgWidth = int(DisplaySize[1] / MaxNumOfImages)
    #factor = imgWidth / Images[0].shape[1]
    for i in range(min(NumberOfImages, MaxNumOfImages)):
        #img = cv2.resize(Images[i], (int(imgWidth * 0.9), int(Images[0].shape[0] * factor * 0.9)))
        img = cv2.resize(Images[i], (ImageSize, ImageSize))
        img_bordered = cv2.copyMakeBorder(src=img, top=BorderWidth, bottom=BorderWidth, left=BorderWidth, right=BorderWidth, value=(255,255,255), borderType=cv2.BORDER_CONSTANT) 
        if i == 0:
            Images4Display = img_bordered
        else:
            #margin = np.zeros([int(Images[0].shape[0]*factor*0.9)+2*BorderWidth,50,3], dtype=np.uint8)
            margin = np.zeros([ImageSize+2*BorderWidth, SeperationWidth, 3], dtype=np.uint8)
            margin[:,:,2] = 193
            #margin[:,:,1] = 0
            #margin[:,:,0] = 0
            Images4Display = cv2.hconcat([Images4Display, margin, img_bordered])

    imgbytes = cv2.imencode('.png', Images4Display)[1].tobytes()                 # Convert the images to a format that PySimpleGUI can display
    return imgbytes

    '''
    elif NumberOfImages == 5:
        imgWidth = int(DisplaySize[1] / 3)
        hFactor = imgWidth / Images[0].shape[1]
        imgHeight = int(DisplaySize[0] / 2)
        vFactor = imgHeight / Images[0].shape[0]
        factor = min(hFactor, vFactor)
        if factor == hFactor:
            width = imgWidth
            height = int(Images[0].shape[0] * factor)
        else:
            width = int(Images[0].shape[1] * factor)
            height = imgHeight

        img1 = cv2.resize(Images[0], (width, height))
        img2 = cv2.resize(Images[1], (width, height))
        img3 = cv2.resize(Images[2], (width, height))
        img4 = cv2.resize(Images[3], (width, height))
        img5 = cv2.resize(Images[4], (width, height))
        img6 = np.zeros([height, width,3],dtype=np.uint8)
        img6.fill(255) # or img[:] = 255

        Images4Display = cv2.vconcat([cv2.hconcat([img1, img2, img3]), cv2.hconcat([img4, img5, img6])])
    '''


### Global variables ###
Images = []
Images4Display = []
DisplaySize = np.array([int((1920 - 150)/3), 1920 - 150])
NumberOfImages = 0
MaxNumOfImages = 3
ImageSize = int((GetSystemMetrics(0) * 0.75)/3)
SeperationWidth = int((GetSystemMetrics(0) * 0.1)/3)
BorderWidth = 5
Debug = False
Paused = False


### Window layout ###
sg.theme("DarkRed1")
Shamir_Logo = get_full_path('ShamirNewLogo2.png')
play_pause_icon = get_full_path('PlayPause.png')
colors = (sg.theme_background_color(), sg.theme_background_color())

if Debug:
    leftLayout = sg.Frame(layout=[
        [sg.Radio('Camera', 'direction', size=(12, 1), key="-Camera-", enable_events=True)],      
        [sg.Radio('Mirror', 'direction', size=(12, 1), key="-Mirror-", enable_events=True, default=True)],      
        [sg.Button("CAPTURE", size=(12, 2), font=('Helvetica', 12, 'bold'), button_color=('darkred', 'salmon'), key="-CAPTURE-")],
        [sg.Button('', image_filename=play_pause_icon, button_color=colors, image_size=(100, 100), image_subsample=3, key="-PLAY_PAUSE-", border_width=0, pad=(5, 10))],
        [sg.Button("Recapture", size=(12, 2), font=('Helvetica', 12, 'bold'), key="-RECAPTURE-")],
        [sg.Button("Approve and\nContinue", size=(12, 2), font=('Helvetica', 12, 'bold'), key="-CONTINUE-")],
        [sg.Button("Clear images", size=(12, 2), font=('Helvetica', 12, 'bold'), key="-CLEAR_IMAGES-")],
        [sg.Button("Exit", size=(12, 2), font=('Helvetica', 12, 'bold'), key="-EXIT-")],
        [sg.VStretch()],
        [sg.Image(Shamir_Logo, subsample=2)]],
        title='', size=(200, 1200), vertical_alignment='top', border_width=0)

    rightLayout = [
        [sg.Image(filename='', size=(ImageSize, ImageSize), key='-VIDEO-'),
         sg.Multiline(default_text='', size=(114,26), key="-OUTPUT-", reroute_cprint=True, no_scrollbar=True, border_width=8, pad=(57, 0))],
        [sg.Image(filename='', size=(ImageSize, GetSystemMetrics(0) * 0.75), key='-IMAGES-')]
    ]
else:
    leftLayout = sg.Frame(layout=[
        [sg.Radio('Mirror', 'direction', size=(12, 1), key="-Mirror-", default=True, visible=False)],      
        [sg.Button("CAPTURE", size=(12, 2), font=('Helvetica', 12, 'bold'), button_color=('darkred', 'salmon'), key="-CAPTURE-")],
        [sg.Button("Recapture", size=(12, 2), font=('Helvetica', 12, 'bold'), key="-RECAPTURE-")],
        [sg.Button("Approve and\nContinue", size=(12, 2), font=('Helvetica', 12, 'bold'), key="-CONTINUE-")],
        [sg.Button("Clear images", size=(12, 2), font=('Helvetica', 12, 'bold'), key="-CLEAR_IMAGES-")],
        [sg.Button("Exit", size=(12, 2), font=('Helvetica', 12, 'bold'), key="-EXIT-")],
        [sg.VStretch()],
        [sg.Image(Shamir_Logo, subsample=2)]],
        title='', size=(200, 1200), vertical_alignment='top', border_width=0)
    
    rgb_values = (193, 0, 0)
    backgroundColor = '#{:02x}{:02x}{:02x}'.format(*rgb_values)
    rightLayout = [
        [sg.Image(filename='', size=(ImageSize, ImageSize), key='-VIDEO-'),
         sg.Multiline(default_text='', size=(32,8), key="-OUTPUT-", reroute_cprint=True, no_scrollbar=True,
                      font=('Courier 40 bold'), justification='c', text_color='white', background_color=backgroundColor, border_width=-1, pad=(57, 0))],
        [sg.Image(filename='', size=(ImageSize, GetSystemMetrics(0) * 0.75), key='-IMAGES-')]
    ]

layout = [[leftLayout, sg.Column(rightLayout)]]

window = sg.Window(
    "SmartMi",
    layout,
    finalize=True,
    font=("Helvetica", 12),
)
window.Maximize()


# Initialize MediaPipe Face Detection
mp_face = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face.FaceDetection(min_detection_confidence=0.2)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)


### Main code ###
vid = cv2.VideoCapture(1)

with open('Frames\\3_L.txt', 'r') as file:
    lines = file.readlines()
coordinates = [line.strip().split(',') for line in lines]
coordinates = [[int(coord[0]), int(coord[1])] for coord in coordinates]
leftContour = np.array(coordinates)

with open('Frames\\3_R.txt', 'r') as file:
    lines = file.readlines()
coordinates = [line.strip().split(',') for line in lines]
coordinates = [[int(coord[0]), int(coord[1])] for coord in coordinates]
rightContour = np.array(coordinates)

leftPupil = np.array([np.mean(leftContour[:,0]), np.mean(leftContour[:,1])])
rightPupil = np.array([np.mean(rightContour[:,0]), np.mean(rightContour[:,1])])
symmetryLine = np.array([[int((leftPupil[0]+rightPupil[0])/2)-1, int((leftPupil[1]+rightPupil[1])/2)+20], [int((leftPupil[0]+rightPupil[0])/2), int((leftPupil[1]+rightPupil[1])/2)-20]])
lenspair = Lenspair(leftContour, rightContour, symmetryLine, leftPupil, rightPupil)
while True:
    ret, frame = vid.read(10)
    diff = frame.shape[1] - frame.shape[0]
    if diff > 0:
        frame = frame[:,int(diff/2+1):int(-diff/2),:]
    elif diff < 0:
        frame = frame[int(diff/2+1):int(-diff/2),:,:]
    frame = cv2.resize(frame, (ImageSize, ImageSize))
    frame = cv2.copyMakeBorder(src=frame, top=BorderWidth, bottom=BorderWidth, left=BorderWidth, right=BorderWidth, value=(255,255,255), borderType=cv2.BORDER_CONSTANT) 

    event, values = window.read(timeout=20)
    if not Paused:
        if values["-Mirror-"]:
            frame = cv2.flip(frame, 1) 
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Face Detection
        results_detection = face_detection.process(rgb_frame)
        if results_detection.detections is None:
            window["-OUTPUT-"].update('No face detected. Please look straight into the camera.')
        elif len(results_detection.detections) > 1:
            window["-OUTPUT-"].update('Multiple faces detected. Please move other people out of the frame.')
        else:
            window["-OUTPUT-"].update('Please look straight into the camera and press CAPTURE button.')

            # Process the frame with MediaPipe Face Mesh
            results_mesh = face_mesh.process(rgb_frame)

            if Debug:
                frameDebugMode = frame.copy()
                
                # Draw face landmarks on the image with smaller points
                if results_mesh.multi_face_landmarks:
                    face_landmarks = results_mesh.multi_face_landmarks[0]
                    faceFeatures = FaceFeatures(face_landmarks, frame.shape)
                    if np.mod(NumberOfImages, 3) == 0:
                        frameColor = (170,50,100)
                    elif np.mod(NumberOfImages, 3) == 1:
                        frameColor = (50,100,170)
                    else:
                        frameColor = (100,170,50)

                    cv2.polylines(frameDebugMode, [np.array(lenspair.transformedLeftContour).astype(np.int32)], False, frameColor, 2)
                    cv2.polylines(frameDebugMode, [np.array(lenspair.transformedRightContour).astype(np.int32)], False, frameColor, 2)

                    draw_landmark_index(frameDebugMode, face_landmarks, faceFeatures, 1)
                    #mp_drawing.draw_landmarks(
                    #     frameDebugMode, face_landmarks,
                    #     mp_face_mesh.FACEMESH_NOSE | mp_face_mesh.FACEMESH_LEFT_EYEBROW | mp_face_mesh.FACEMESH_RIGHT_EYEBROW,
                    #     landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=0),
                    #     connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1))

                    if cv2.pointPolygonTest(leftContour, (face_landmarks.landmark[473].x, face_landmarks.landmark[473].y), False) < 1 or \
                        cv2.pointPolygonTest(rightContour, (face_landmarks.landmark[468].x, face_landmarks.landmark[468].y), False) < 1:
                        window["-OUTPUT-"].update('Lenspair is not set properly on patient face.')
                    else:
                        score = Score(faceFeatures, lenspair, 0)

                        window['-OUTPUT-'].update(value='')
                        sg.cprint('Frame width ratio = {fw0}'.format(fw0=score.frameWidthRatio), font='Courier 10')
                        sg.cprint('Frame width score = {fw}'.format(fw=score.frameWidthScore), font='Courier 12 bold')
                        sg.cprint('')
                        sg.cprint('Left eyebrow score = {em0}'.format(em0=score.leftEyebrowScore), font='Courier 10')
                        sg.cprint('Right eyebrow score = {em1}'.format(em1=score.rightEyebrowScore), font='Courier 10')
                        sg.cprint('Eyebrows match score = {em}'.format(em=score.eyebrowsMatchScore), font='Courier 12 bold')
                        sg.cprint('')
                        sg.cprint('Left cheek line score = {lcl0}'.format(lcl0=score.leftCheekLineScore), font='Courier 10')
                        sg.cprint('Right cheek line score = {lcl1}'.format(lcl1=score.rightCheekLineScore), font='Courier 10')
                        sg.cprint('Lower cheek line score = {lcl}'.format(lcl=score.lowerCheekLineScore), font='Courier 12 bold')
                        sg.cprint('')
                        sg.cprint('Circularity panelty = {fs0}'.format(fs0=score.circularityPanelty), font='Courier 10')
                        sg.cprint('Face aspect ratio = {fs1}'.format(fs1=score.faceAspectRatio), font='Courier 10')
                        sg.cprint('Face circularity score = {fs2}'.format(fs2=score.face_circularityScore), font='Courier 10')
                        sg.cprint('Frame shape score = {fs}'.format(fs=score.frameShapeScore), font='Courier 12 bold')
                        sg.cprint('')
                        sg.cprint('Total score = {t}'.format(t=score.totalScore), font='Courier 12 bold')

                        #scoreText = edit_score_text(score)
                        #window["-OUTPUT-"].update(scoreText)
        if Debug:
            imgbytes = cv2.imencode('.png', frameDebugMode)[1].tobytes()                 # Convert the frame to a format that PySimpleGUI can display
        else:
            imgbytes = cv2.imencode('.png', frame)[1].tobytes()                 # Convert the frame to a format that PySimpleGUI can display
        window['-VIDEO-'].update(data=imgbytes)

    if event == "-EXIT-" or event == sg.WIN_CLOSED:
        break

    elif event == "-CAPTURE-":
        if Paused:
            window["-OUTPUT-"].update('Please approve image to continue capturing.')
            continue
        
        face = extract_face(frame)
        if not len(face):
            continue

        if NumberOfImages >= MaxNumOfImages:
            Images.pop(0)
        NumberOfImages = NumberOfImages + 1
        
        if Debug:
            score = Score(faceFeatures, lenspair, 1)
            cv2.putText(face, f'{score.totalScore}', (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (155, 0, 0), 1, cv2.LINE_AA)
        Images.append(face)
        imgbytes = set_images_for_display()
        window['-IMAGES-'].update(data=imgbytes)
        window["-OUTPUT-"].update('Image captured.')
        Paused = True

    elif event == "-RECAPTURE-":
        if Paused:
            if NumberOfImages > 0:
                Images.pop(-1)
                NumberOfImages = NumberOfImages - 1
            imgbytes = set_images_for_display()
            window['-IMAGES-'].update(data=imgbytes)
            Paused = not Paused
        
    elif Debug and event == "-PLAY_PAUSE-":
        Paused = not Paused

    if event == "-CONTINUE-":
        if Paused:
            Paused = not Paused

    elif event == "-CLEAR_IMAGES-":
        if len(Images) == 0:
            continue

        Images = []
        NumberOfImages = 0
        imgbytes = set_images_for_display()
        window['-IMAGES-'].update(data=imgbytes)

vid.release()
window.close()
