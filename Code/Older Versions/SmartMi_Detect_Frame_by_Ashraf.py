import numpy as np
import cv2
import PySimpleGUI as sg
import os.path
import mediapipe as mp
import time
from win32api import GetSystemMetrics
from SmartMiObjects import FaceFeatures, Lenspair, Score, State
from datetime import datetime
import subprocess
import json


### Description ###
# SmartMi enables to display multiple images taken from a video stream simultaneously, in order to compare shown features.


### Functions ###
def create_splash_screen_layout_window():
    global state
    state = State.SPLASH
    window = sg.Window("SmartMi", create_splash_screen_layout(), icon=os.path.join(IconsPath,'TitleLogo.ico'), element_justification='center', finalize=True, no_titlebar=True)
    window.Maximize()

    imgbytes = cv2.imencode('.png', StartIcon)[1].tobytes()                 # Convert the images to a format that PySimpleGUI can display
    window['-START-'].draw_image(data=imgbytes, location=(0, 0))
    return window

def create_app_layout_window():
    global state
    state = State.APP
    window = sg.Window("SmartMi", create_app_layout(), icon=os.path.join(IconsPath,'TitleLogo.ico'), finalize=True, no_titlebar=True)
    window.Maximize()
    #for i in range(MaxNumOfImages):
    #    window[f'-GRAPH{i}-'].draw_image(data=ClearImage, location=(0, 0))
    return window

def create_splash_screen_layout():
    layout = [
        [sg.Image(Shamir_Logo, size=(ImageSize, ImageSize), key='-SPLASH_SCREEN_ICON-')],
        [sg.Graph((StartIcon.shape[1], StartIcon.shape[0]), (0, StartIcon.shape[0]), (StartIcon.shape[1], 0), key='-START-', enable_events=True, visible=True)]
    ]
    return layout

def create_app_layout():
    backgroundColor = '#{:02x}{:02x}{:02x}'.format(*BackgroundColor)
    horizontalPedding = 40
    buttonsFrame = sg.Frame(layout=[
        [sg.Radio('Mirror', 'direction', size=(12, 1), key="-Mirror-", default=True, visible=False)],      
        [sg.Button("CAPTURE", size=(12, 2), font=ButtonFont, button_color=('darkred', 'salmon'), key="-CAPTURE-", pad=(0, 20))],
        [sg.Button("Clear images", size=(12, 2), font=ButtonFont, key="-CLEAR_IMAGES-")],
        [sg.Button("Close", size=(12, 2), font=ButtonFont, key="-CLOSE-")],
        [sg.VStretch()],
        [sg.Image(Shamir_Logo, subsample=2)]],
        title='', size=(ImageSize+2*BorderWidth, ImageSize+2*BorderWidth), vertical_alignment='top', element_justification='center', border_width=0, pad=(horizontalPedding, 0))

    if Debug:
        layout = [
            [buttonsFrame, sg.Image(filename='', size=(ImageSize, ImageSize), key='-VIDEO-', pad=(120, 30)),
                sg.Multiline(default_text='', size=(40,27), key="-OUTPUT_DEBUG-", reroute_cprint=True, no_scrollbar=True,
                            font='Helvetica 12', justification='c', text_color='white', background_color=backgroundColor, border_width=1, pad=(int(horizontalPedding/4), 0)),
                sg.Multiline(default_text='', size=(70,14), key="-OUTPUT-", no_scrollbar=True,
                            font='Courier 20 bold', justification='c', text_color='white', background_color=backgroundColor, border_width=-1, pad=(int(horizontalPedding/2), 0))],
            [sg.Graph((ImageSize+2*BorderWidth, ImageSize+2*BorderWidth), (0, ImageSize+2*BorderWidth), (ImageSize+2*BorderWidth, 0), key='-GRAPH0-', enable_events=True, visible=True, pad=(horizontalPedding, 0)),
                sg.Graph((ImageSize+2*BorderWidth, ImageSize+2*BorderWidth), (0, ImageSize+2*BorderWidth), (ImageSize+2*BorderWidth, 0), key='-GRAPH1-', enable_events=True, visible=True, pad=(120, 0)),
                sg.Graph((ImageSize+2*BorderWidth, ImageSize+2*BorderWidth), (0, ImageSize+2*BorderWidth), (ImageSize+2*BorderWidth, 0), key='-GRAPH2-', enable_events=True, visible=True, pad=(horizontalPedding, 0))]
        ]
    else:
        layout = [
            [buttonsFrame, sg.Image(filename='', size=(ImageSize, ImageSize), key='-VIDEO-', pad=(120, 30)),
                sg.Multiline(default_text='', size=(20,8), key="-OUTPUT-", no_scrollbar=True,
                            font='Courier 30 bold', justification='c', text_color='white', background_color=backgroundColor, border_width=-1, pad=(horizontalPedding, 0))],
            [sg.Graph((ImageSize+2*BorderWidth, ImageSize+2*BorderWidth), (0, ImageSize+2*BorderWidth), (ImageSize+2*BorderWidth, 0), key='-GRAPH0-', enable_events=True, visible=True, pad=(horizontalPedding, 0)),
                sg.Graph((ImageSize+2*BorderWidth, ImageSize+2*BorderWidth), (0, ImageSize+2*BorderWidth), (ImageSize+2*BorderWidth, 0), key='-GRAPH1-', enable_events=True, visible=True, pad=(120, 0)),
                sg.Graph((ImageSize+2*BorderWidth, ImageSize+2*BorderWidth), (0, ImageSize+2*BorderWidth), (ImageSize+2*BorderWidth, 0), key='-GRAPH2-', enable_events=True, visible=True, pad=(horizontalPedding, 0))]
        ]

    return layout

def draw_landmark_index(image, landmarks, faceFeatures, score):
    #S = [127, 356, 10, 152, 193, 417, 4, 101, 36, 330, 266, 108, 151, 337]
    S = [127, 356, 10, 152, 193, 417]
    for i in S:
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
    Face circularity  = {fs2}
    Frame shape  = {fs}

    Total score = {t}\
    '''.format(fw0=score.frameWidthRatio, fw=score.frameWidth, em0=score.leftEyebrow, em1=score.rightEyebrow, em=score.eyebrowsMatch,\
               lcl0=score.leftCheekLine, lcl1=score.rightCheekLine, lcl=score.lowerCheekLine,\
                fs0=score.circularityPanelty, fs1=score.faceAspectRatio, fs2=score.face_circularity, fs=score.frameShape, t=score.total)

    scoreText = edit_rows(scoreText)
    return scoreText

def plot_features_debug_mode(face, lenspair, face_landmarks, faceFeatures):
    imgDebugMode = face.copy()
    frameColor = (255,255,255)
    #cv2.polylines(imgDebugMode, [np.array(lenspair.transformedLeftContour).astype(np.int32)], False, frameColor, 1)
    #cv2.polylines(imgDebugMode, [np.array(lenspair.transformedRightContour).astype(np.int32)], False, frameColor, 1)
    cv2.polylines(imgDebugMode, [np.array(lenspair.leftContour).astype(np.int32)], False, frameColor, 1)
    cv2.polylines(imgDebugMode, [np.array(lenspair.rightContour).astype(np.int32)], False, frameColor, 1)

    draw_landmark_index(imgDebugMode, face_landmarks, faceFeatures, 1)
    #mp_drawing.draw_landmarks(
    #     imgDebugMode, face_landmarks,
    #     mp_face_mesh.FACEMESH_NOSE | mp_face_mesh.FACEMESH_LEFT_EYEBROW | mp_face_mesh.FACEMESH_RIGHT_EYEBROW,
    #     landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=0),
    #     connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1))
    return imgDebugMode

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

def set_clear_image():
    clearImage = np.full((ImageSize, ImageSize, 3), BackgroundColor[::-1], dtype=np.uint8)
    clearImage = cv2.copyMakeBorder(src=clearImage, top=BorderWidth, bottom=BorderWidth, left=BorderWidth, right=BorderWidth, value=BackgroundColor[::-1], borderType=cv2.BORDER_CONSTANT) 
    imgbytes = cv2.imencode('.png', clearImage)[1].tobytes()                 # Convert the images to a format that PySimpleGUI can display
    return imgbytes

def display_detailed_score(score):
    window['-OUTPUT_DEBUG-'].update(value='')
    sg.cprint('Frame width ratio = {}'.format(score.frameWidthRatio), font='Courier 10')
    sg.cprint('Frame width score = {}'.format(score.frameWidth), font='Courier 12 bold')
    sg.cprint('')
    sg.cprint('Left eyebrow score = {}'.format(score.leftEyebrow), font='Courier 10')
    sg.cprint('Right eyebrow score = {}'.format(score.rightEyebrow), font='Courier 10')
    sg.cprint('Eyebrows match score = {}'.format(score.eyebrowsMatch), font='Courier 12 bold')
    sg.cprint('')
    sg.cprint('Left cheek line score = {}'.format(score.leftCheekLine), font='Courier 10')
    sg.cprint('Right cheek line score = {}'.format(score.rightCheekLine), font='Courier 10')
    sg.cprint('Lower cheek line score = {}'.format(score.lowerCheekLine), font='Courier 12 bold')
    sg.cprint('')
    sg.cprint('Circularity panelty = {}'.format(score.circularityPanelty), font='Courier 10')
    sg.cprint('Face aspect ratio = {}'.format(score.faceAspectRatio), font='Courier 10')
    sg.cprint('Face circularity score = {}'.format(score.face_circularity), font='Courier 10')
    sg.cprint('Frame shape score = {}'.format(score.frameShape), font='Courier 12 bold')
    sg.cprint('')
    sg.cprint('DBL width ratio = {}'.format(score.DBLWidthRatio), font='Courier 10')
    sg.cprint('DBL score = {}'.format(score.DBL), font='Courier 12 bold')
    sg.cprint('')
    sg.cprint('Frame color diff = {}'.format(score.colorDiff), font='Courier 10')
    sg.cprint('Frame color score = {}'.format(score.frameColor), font='Courier 12 bold')
    sg.cprint('')
    sg.cprint('Total score = {}'.format(score.total), font='Courier 12 bold')

    #scoreText = edit_score_text(score)
    #window["-OUTPUT_DEBUG-"].update(scoreText)

def extract_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3, minSize=(30, 30))

    #print("Found {0} faces!".format(len(faces)))
    if not len(faces):
        return []

    #for (x, y, w, h) in faces:
    #    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    (x, y, w, h) = faces[0]
    d = max(w, h)

    if x-int(d/2) <= 0 or x+int(3*d/2) >= frame.shape[1] or y-int(d/2) <= 0 or y+int(3*d/2) >= frame.shape[0]:
        raise Exception("Please move towards the center of the frame.")
    minX = max(0, x-int(d/2))
    maxX = min(frame.shape[1], x+int(3*d/2))
    minY = max(0, y-int(d/2))
    maxY = min(frame.shape[0], y+int(3*d/2))
    #frame = cv2.circle(frame, (x+int(d/2),y+int(d/2)), 5, (0, 0, 255)) 
    croppedFrame = frame[minY:maxY, minX:maxX]

    return croppedFrame

def get_index_for_display():
    global IndicesQueue

    if NumOfImages >= MaxNumOfImages:
        idx = IndicesQueue[0]
        IndicesQueue.pop(0)
    else:
        if len(Images4Display['0']) > 0:
            if len(Images4Display['1']) > 0:
                idx = 2
            else:
                idx = 1
        else:
            idx = 0

    IndicesQueue.append(idx)
    return idx

def set_image_for_display(img, i):
    global Images4Display
    img = cv2.resize(img, (ImageSize, ImageSize))
    imgBordered = cv2.copyMakeBorder(src=img, top=BorderWidth, bottom=BorderWidth, left=BorderWidth, right=BorderWidth, value=(255,255,255), borderType=cv2.BORDER_CONSTANT) 
    imgbytes = cv2.imencode('.png', imgBordered)[1].tobytes()                 # Convert the images to a format that PySimpleGUI can display
    #Images4Display.append(imgbytes)
    Images4Display[f'{i}'] = imgbytes

def clear_images():
    global NumOfImages, IndicesQueue, Images4Display, window
    if NumOfImages == 0:
        return

    NumOfImages = 0
    IndicesQueue = []
    for i in range(MaxNumOfImages):
        Images4Display[f'{i}'] = []
        window[f'-GRAPH{i}-'].draw_image(data=ClearImage, location=(0, 0))

def get_frame_contours(frame):
    # Convert the frame to bytes
    success = cv2.imwrite('face.jpg', frame)
    if not success:
        raise Exception("Could not save face image.")
    
    relative_path = os.path.join('.', 'SparkAI.API.exe')
    parameters = ["face.jpg"]  # List of parameters

    # Combine the executable and parameters into a single command
    command = [relative_path] + parameters

    # Run the command
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception("Failed to extract frame contours.")

    leftContour, rightContour = parse_frame_contour("face_contour_points.json")
    return leftContour, rightContour

def parse_frame_contour(jsonFile): 
    # Read the JSON file
    with open(jsonFile, 'r') as file:
        contours = json.load(file)
    
    # Create lists for RightContour and LeftContour
    leftContour = [[point["X"], point["Y"]] for point in contours["LeftContour"]]
    rightContour = [[point["X"], point["Y"]] for point in contours["RightContour"]]
    leftContour = np.array(leftContour)
    rightContour = np.array(rightContour)
    return leftContour, rightContour


### Global variables ###
Images4Display = {"0": [], "1": [], "2": []}
MaxNumOfImages = len(Images4Display)
NumOfImages = 0
DisplaySize = np.array([int((1920 - 150)/MaxNumOfImages), 1920 - 150])
ImageSize = int((GetSystemMetrics(0) * 0.75)/MaxNumOfImages)
ButtonFont = ('Helvetica 12 bold')
BorderWidth = 5
BackgroundColor = (193,0,0)
ClearImage = set_clear_image()
IndicesQueue = []
Path = os.path.dirname(os.path.abspath(__file__))
IconsPath = os.path.join(os.path.abspath(os.path.join(Path, os.pardir)), 'Icons')
vid = []
state = State.SPLASH
FaceBadLocated = False
Debug = True


### Window layout ###
sg.theme("DarkRed1")
Shamir_Logo = os.path.join(IconsPath, 'ShamirNewLogo2.png')
StartIcon = cv2.imread(os.path.join(IconsPath, 'Start.png'))
window = create_splash_screen_layout_window()


# Initialize MediaPipe Face Detection
mp_face = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face.FaceDetection(min_detection_confidence=0.2)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)


### Main code ###
#a = datetime.now()
vid = cv2.VideoCapture(0)
#b = datetime.now()
#print(b - a)

while True:
    event, values = window.read(timeout=20)
    if event == "-EXIT-" or event == sg.WIN_CLOSED:
        break
    elif event == "-START-":
        window.close()
        window = create_app_layout_window()
        event, values = window.read(timeout=20)
    elif event == "-CLOSE-":
        clear_images()
        window.close()
        window = create_splash_screen_layout_window()
        event, values = window.read(timeout=20)

    if state != State.SPLASH:
        ret, frame = vid.read(10)
        diff = frame.shape[1] - frame.shape[0]
        if diff > 0:
            frame = frame[:,int(diff/2+1):int(-diff/2),:]
        elif diff < 0:
            frame = frame[int(diff/2+1):int(-diff/2),:,:]
        frame = cv2.resize(frame, (ImageSize, ImageSize))
        frame = cv2.copyMakeBorder(src=frame, top=BorderWidth, bottom=BorderWidth, left=BorderWidth, right=BorderWidth, value=(255,255,255), borderType=cv2.BORDER_CONSTANT) 

        if values["-Mirror-"]:
            frame = cv2.flip(frame, 1) 
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Face Detection
        results_detection = face_detection.process(rgb_frame)

        if FaceBadLocated:
            time.sleep(2)
            FaceBadLocated = False
        elif results_detection.detections is None:
            window["-OUTPUT-"].update('No face detected. Please look straight into the camera.')
        elif len(results_detection.detections) > 1:
            window["-OUTPUT-"].update('Multiple faces detected. Please move other people out of the frame.')
        else:
            window["-OUTPUT-"].update('Please look straight into the camera and press CAPTURE button.')

        imgbytes = cv2.imencode('.png', frame)[1].tobytes()                 # Convert the frame to a format that PySimpleGUI can display
        window['-VIDEO-'].update(data=imgbytes)

        if event == "-CAPTURE-":
            try:
                face = extract_face(frame)
            except Exception as err:
                window["-OUTPUT-"].update(err)
                FaceBadLocated = True
                continue
            
            if not len(face):
                continue
                                    
            try:
                leftContour, rightContour = get_frame_contours(face)
            except Exception as err:
                window["-OUTPUT-"].update(err)
                continue
            if len(leftContour) == 0 or len(rightContour) == 0:
                window["-OUTPUT-"].update('No glasses frame detected.')
                continue

            leftPupil = np.array([np.mean(leftContour[:,0]), np.mean(leftContour[:,1])])
            rightPupil = np.array([np.mean(rightContour[:,0]), np.mean(rightContour[:,1])])
            symmetryLine = np.array([[int((leftPupil[0]+rightPupil[0])/2)-1, int((leftPupil[1]+rightPupil[1])/2)+20], [int((leftPupil[0]+rightPupil[0])/2), int((leftPupil[1]+rightPupil[1])/2)-20]])
            lenspair = Lenspair(leftContour, rightContour, symmetryLine, leftPupil, rightPupil)

            ## Face detection
            rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            results_mesh = face_mesh.process(rgb_face)

            if results_mesh.multi_face_landmarks:
                face_landmarks = results_mesh.multi_face_landmarks[0]
                faceFeatures = FaceFeatures(rgb_face, face_landmarks, face.shape)

            if Debug:
                score = Score(faceFeatures, lenspair, 1)
                display_detailed_score(score)
                cv2.putText(face, f'{score.total}', (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (155, 0, 0), 1, cv2.LINE_AA)
                faceDebugMode = plot_features_debug_mode(face, lenspair, face_landmarks, faceFeatures)
                imgbytes = cv2.imencode('.png', faceDebugMode)[1].tobytes()                 # Convert the frame to a format that PySimpleGUI can display
                idx = get_index_for_display()
                set_image_for_display(faceDebugMode, idx)
            else:
                score = Score(faceFeatures, lenspair, 0)
                cv2.putText(face, f'{score.total}', (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (155, 0, 0), 1, cv2.LINE_AA)
                imgbytes = cv2.imencode('.png', face)[1].tobytes()                 # Convert the frame to a format that PySimpleGUI can display
                idx = get_index_for_display()
                set_image_for_display(face, idx)

            NumOfImages = min(NumOfImages + 1, MaxNumOfImages)

            for i in range(MaxNumOfImages):
                window[f'-GRAPH{i}-'].draw_image(data=Images4Display[f'{i}'], location=(0, 0))
            window["-OUTPUT-"].update('Image captured.')
            
        elif event == "-CLEAR_IMAGES-":
            clear_images()

        for i in range(MaxNumOfImages):
            if event == f'-GRAPH{i}-':
                if len(Images4Display[f'{i}']) > 0:
                    ans = sg.popup('Are you sure you want to clear the image?', title='Clear image', background_color='lightgray', text_color='darkred',
                                custom_text=('Yes', 'No'), icon=os.path.join(IconsPath,'TitleLogo.ico'))
                    if ans == 'Yes':
                        Images4Display[f'{i}'] = []
                        window[f'-GRAPH{i}-'].draw_image(data=ClearImage, location=(0, 0))
                        NumOfImages = NumOfImages - 1
                        IndicesQueue.remove(i)
                continue
            
vid.release()
window.close()
