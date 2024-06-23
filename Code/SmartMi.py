import numpy as np
import cv2
import PySimpleGUI as sg
import os.path
import mediapipe as mp
from PIL import Image
import io
from win32api import GetSystemMetrics
from Classes import FaceFeatures, Lenspair, Score, State
import subprocess
import json
import sqlite3

import threading
import time
from queue import Queue

### Spark4 Monitor ###
def monitor_new_measurements(db_path, queue):
    while True:
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute('SELECT ColorTop FROM new_measurements order by id desc LIMIT 1')
            new_rows = cursor.fetchall()

            for row in new_rows:
                queue.put(row)  # Send data to the main thread via queue
                cursor.execute('DELETE FROM new_measurements')
        finally:
            conn.commit()
            conn.close()
            time.sleep(1)  # Adjust the sleep time as necessary


def start_monitor_thread(db_path, callback):
    queue = Queue()
    
    # Function to process data from the queue
    def process_queue():
        while True:
            row = queue.get()
            callback(row)  # Call the provided callback with the row data

    monitor_thread = threading.Thread(target=monitor_new_measurements, args=(db_path, queue))
    monitor_thread.daemon = True  # Daemonize thread to exit with the main program
    monitor_thread.start()

    process_thread = threading.Thread(target=process_queue)
    process_thread.daemon = True
    process_thread.start()

def my_callback(row):
    try:
        # Convert the binary data to an image
        raw_image = Image.open(io.BytesIO(row[0]))
        image = cv2.cvtColor(np.array(raw_image), cv2.COLOR_RGB2BGR)

        # Process the frame with MediaPipe Face Detection
        results_detection = face_detection.process(image)
        leftContour, rightContour = get_frame_contours(image)
    except Exception as err:
        window["-OUTPUT-"].update(err)
        return
                                        
    leftPupil = np.array([np.mean(leftContour[:,0]), np.mean(leftContour[:,1])])
    rightPupil = np.array([np.mean(rightContour[:,0]), np.mean(rightContour[:,1])])
    symmetryLine = np.array([[int((leftPupil[0]+rightPupil[0])/2)-1, int((leftPupil[1]+rightPupil[1])/2)+20], [int((leftPupil[0]+rightPupil[0])/2), int((leftPupil[1]+rightPupil[1])/2)-20]])
    lenspair = Lenspair(leftContour, rightContour, symmetryLine, leftPupil, rightPupil)

    ## Face detection
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(rgb_image)

    if results_mesh.multi_face_landmarks:
        face_landmarks = results_mesh.multi_face_landmarks[0]
        faceFeatures = FaceFeatures(rgb_image, face_landmarks, image.shape)

    #x, y, w, h = detect_face_location(image)
    #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 4)
    #cv2.circle(image, (x, y), 10, (0, 255, 255), -1)
    
    if Debug:
        score = Score(faceFeatures, lenspair, 1)
        display_detailed_score(score)
        imageDebugMode = plot_features_debug_mode(image, lenspair, face_landmarks, faceFeatures)
        #faceDebugMode = centerize_face(imageDebugMode)

        cv2.putText(imageDebugMode, f'{score.total}', (100,100), cv2.FONT_HERSHEY_SIMPLEX, 4, (155, 0, 0), 5, cv2.LINE_AA)
        imgbytes = cv2.imencode('.png', imageDebugMode)[1].tobytes()                 # Convert the frame to a format that PySimpleGUI can display
        idx = get_index_for_display()
        set_image_for_display(imageDebugMode, idx)
    else:
        score = Score(faceFeatures, lenspair, 0)
        #face = centerize_face(image)

        cv2.putText(image, f'{score.total}', (100,100), cv2.FONT_HERSHEY_SIMPLEX, 4, (155, 0, 0), 5, cv2.LINE_AA)
        imgbytes = cv2.imencode('.png', image)[1].tobytes()                 # Convert the frame to a format that PySimpleGUI can display
        idx = get_index_for_display()
        set_image_for_display(image, idx)

    global NumOfImages
    global MaxNumOfImages
    NumOfImages = min(NumOfImages + 1, MaxNumOfImages)

    for i in range(MaxNumOfImages):
        window[f'-GRAPH{i}-'].draw_image(data=Images4Display[f'{i}'], location=(0, 0))
    window["-OUTPUT-"].update('Image was loaded.')


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
        [sg.Image(Shamir_Logo, size=(ImageSize[1], ImageSize[1]), key='-SPLASH_SCREEN_ICON-')],
        [sg.Graph((StartIcon.shape[1], StartIcon.shape[0]), (0, StartIcon.shape[0]), (StartIcon.shape[1], 0), key='-START-', enable_events=True, visible=True)]
    ]
    return layout

def create_app_layout():
    backgroundColor = '#{:02x}{:02x}{:02x}'.format(*BackgroundColor)
    horizontalPedding = 40
    buttonsFrame = sg.Frame(layout=[
        [sg.Text("Enter image ID:"), sg.InputText(size=(20,1), key="-IMAGE_ID-")],
        [sg.Button("Get Image", size=(12, 2), font=ButtonFont, button_color=('darkred', 'salmon'), key="-GET_IMAGE-", pad=(0, 10))],
        [sg.Button("Clear images", size=(12, 2), font=ButtonFont, key="-CLEAR_IMAGES-")],
        [sg.Button("Close", size=(12, 2), font=ButtonFont, key="-CLOSE-")],
        [sg.VStretch()],
        [sg.Image(Shamir_Logo, subsample=2)]],
        title='', size=(350, 350), vertical_alignment='top', element_justification='center', border_width=0, pad=(horizontalPedding, 20))

    if Debug:
        layout = [
            [buttonsFrame, sg.Multiline(default_text='', size=(45,25), key="-OUTPUT_DEBUG-", reroute_cprint=True, no_scrollbar=True,
                            font='Helvetica 10', justification='c', text_color='white', background_color=backgroundColor, border_width=1, pad=(120, 20)),
                sg.Multiline(default_text='', size=(45,8), key="-OUTPUT-", no_scrollbar=True,
                            font='Courier 30 bold', justification='c', text_color='white', background_color=backgroundColor, border_width=-1, pad=(50, 20))],
            [sg.Graph((ImageSize[1]+2*BorderWidth, ImageSize[0]+2*BorderWidth), (0, ImageSize[1]+2*BorderWidth), (ImageSize[0]+2*BorderWidth, 0), key='-GRAPH0-', enable_events=True, visible=True, pad=(horizontalPedding, 0)),
                sg.Graph((ImageSize[1]+2*BorderWidth, ImageSize[0]+2*BorderWidth), (0, ImageSize[1]+2*BorderWidth), (ImageSize[0]+2*BorderWidth, 0), key='-GRAPH1-', enable_events=True, visible=True, pad=(120, 0)),
                sg.Graph((ImageSize[1]+2*BorderWidth, ImageSize[0]+2*BorderWidth), (0, ImageSize[1]+2*BorderWidth), (ImageSize[0]+2*BorderWidth, 0), key='-GRAPH2-', enable_events=True, visible=True, pad=(horizontalPedding, 0))]
        ]
    else:
        layout = [
            [buttonsFrame, sg.Multiline(default_text='', size=(45,8), key="-OUTPUT-", no_scrollbar=True,
                            font='Courier 30 bold', justification='c', text_color='white', background_color=backgroundColor, border_width=1, pad=(horizontalPedding, 0))],
            [sg.Graph((ImageSize[1]+2*BorderWidth, ImageSize[0]+2*BorderWidth), (0, ImageSize[1]+2*BorderWidth), (ImageSize[0]+2*BorderWidth, 0), key='-GRAPH0-', enable_events=True, visible=True, pad=(horizontalPedding, 0)),
                sg.Graph((ImageSize[1]+2*BorderWidth, ImageSize[0]+2*BorderWidth), (0, ImageSize[1]+2*BorderWidth), (ImageSize[0]+2*BorderWidth, 0), key='-GRAPH1-', enable_events=True, visible=True, pad=(120, 0)),
                sg.Graph((ImageSize[1]+2*BorderWidth, ImageSize[0]+2*BorderWidth), (0, ImageSize[1]+2*BorderWidth), (ImageSize[0]+2*BorderWidth, 0), key='-GRAPH2-', enable_events=True, visible=True, pad=(horizontalPedding, 0))]
        ]

    return layout

def draw_landmark_index(image, landmarks, faceFeatures, score, lineWidth):
    #S = [127, 356, 10, 152, 193, 417, 4, 101, 36, 330, 266, 108, 151, 337]
    S = [127, 356, 10, 152, 193, 417]
    for i in S:
        x, y = int(landmarks.landmark[i].x), int(landmarks.landmark[i].y)
        cv2.putText(image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 0), lineWidth, cv2.LINE_AA)
        cv2.circle(image, (x, y), 10, (0, 255, 0), -1)  # Draw a small circle at the landmark position

    cv2.polylines(image, [np.array(faceFeatures.leftEyebrow)], False, (0,0,255), lineWidth)
    cv2.polylines(image, [np.array(faceFeatures.rightEyebrow)], False, (0,0,255), lineWidth)
    cv2.polylines(image, [np.array(faceFeatures.leftUpperCheekLine)], False, (255,255,0), lineWidth)
    cv2.polylines(image, [np.array(faceFeatures.rightUpperCheekLine)], False, (255,255,0), lineWidth)
    cv2.polylines(image, [np.array(faceFeatures.leftLowerCheekLine)], False, (255,255,0), lineWidth)
    cv2.polylines(image, [np.array(faceFeatures.rightLowerCheekLine)], False, (255,255,0), lineWidth)
    cv2.polylines(image, [np.array(faceFeatures.jawLine)], False, (255,0,255), lineWidth)

    if type(score) != int:
        cv2.circle(image, tuple(score.face_circleCenter.astype(np.int32)), int(score.face_circleRadius), (76, 230, 150), lineWidth)
        cv2.circle(image, tuple(score.leftLens_circleCenter.astype(np.int32)), int(score.leftLens_circleRadius), (76, 230, 150), lineWidth)
        cv2.circle(image, tuple(score.rightLens_circleCenter.astype(np.int32)), int(score.rightLens_circleRadius), (76, 230, 150), lineWidth)

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
    lineWidth = 5

    #cv2.polylines(imgDebugMode, [np.array(lenspair.transformedLeftContour).astype(np.int32)], False, frameColor, lineWidth)
    #cv2.polylines(imgDebugMode, [np.array(lenspair.transformedRightContour).astype(np.int32)], False, frameColor, lineWidth)
    cv2.polylines(imgDebugMode, [np.array(lenspair.leftContour).astype(np.int32)], False, frameColor, lineWidth)
    cv2.polylines(imgDebugMode, [np.array(lenspair.rightContour).astype(np.int32)], False, frameColor, lineWidth)

    draw_landmark_index(imgDebugMode, face_landmarks, faceFeatures, 1, lineWidth)
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
    clearImage = np.full((ImageSize[0], ImageSize[1], 3), BackgroundColor[::-1], dtype=np.uint8)
    clearImage = cv2.copyMakeBorder(src=clearImage, top=BorderWidth, bottom=BorderWidth, left=BorderWidth, right=BorderWidth, value=BackgroundColor[::-1], borderType=cv2.BORDER_CONSTANT) 
    imgbytes = cv2.imencode('.png', clearImage)[1].tobytes()                 # Convert the images to a format that PySimpleGUI can display
    return imgbytes

def display_detailed_score(score):
    normalFont = 'Courier 9'
    titleFont = 'Courier 10 bold'
    window['-OUTPUT_DEBUG-'].update(value='')
    sg.cprint('Frame width ratio = {}'.format(score.frameWidthRatio), font=normalFont)
    sg.cprint('Frame width score = {}'.format(score.frameWidth), font=titleFont)
    sg.cprint('')
    sg.cprint('Left eyebrow score = {}'.format(score.leftEyebrow), font=normalFont)
    sg.cprint('Right eyebrow score = {}'.format(score.rightEyebrow), font=normalFont)
    sg.cprint('Eyebrows match score = {}'.format(score.eyebrowsMatch), font=titleFont)
    sg.cprint('')
    sg.cprint('Left cheek line score = {}'.format(score.leftCheekLine), font=normalFont)
    sg.cprint('Right cheek line score = {}'.format(score.rightCheekLine), font=normalFont)
    sg.cprint('Lower cheek line score = {}'.format(score.lowerCheekLine), font=titleFont)
    sg.cprint('')
    sg.cprint('Circularity panelty = {}'.format(score.circularityPanelty), font=normalFont)
    sg.cprint('Face aspect ratio = {}'.format(score.faceAspectRatio), font=normalFont)
    sg.cprint('Face circularity score = {}'.format(score.face_circularity), font=normalFont)
    sg.cprint('Frame shape score = {}'.format(score.frameShape), font=titleFont)
    sg.cprint('')
    sg.cprint('DBL width ratio = {}'.format(score.DBLWidthRatio), font=normalFont)
    sg.cprint('DBL score = {}'.format(score.DBL), font=titleFont)
    sg.cprint('')
    sg.cprint('Frame color diff = {}'.format(score.colorDiff), font=normalFont)
    sg.cprint('Frame color score = {}'.format(score.frameColor), font=titleFont)
    sg.cprint('')
    sg.cprint('Total score = {}'.format(score.total), font=titleFont)

    #scoreText = edit_score_text(score)
    #window["-OUTPUT_DEBUG-"].update(scoreText)

def detect_face_location(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3, minSize=(30, 30))
    if not len(faces):
        return []

    (x, y, w, h) = faces[0]
    return x, y, w, h

def centerize_face(img):
    midY = int(img.shape[0]/2)
    midX = int(img.shape[1]/2)
    if img.shape[0] > img.shape[1]:             # Portrait image
        cropped_image = img[midY-midX:midY+midX, :]
    elif img.shape[0] < img.shape[1]:           # Landscape image
        cropped_image = img[:, midX-midY:midX+midY]
    return cropped_image

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

    img = cv2.resize(img, (ImageSize[1], ImageSize[0]))
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

def execute(db_path, query, params=()):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()    
    try:
        # Execute the query
        cursor.execute(query, params)        
        # Fetch all results from the executed query
        results = cursor.fetchall()        
        return results    
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")    
    finally:
        # Close the cursor and connection
        cursor.close()
        conn.close()

def get_image(imageID):
    results = execute(r'C:\ProgramData\Shamir\Spark4\DB\Spark4.db', 'SELECT ColorTop FROM Measurement AS m JOIN MeasurementSnapshot AS ms ON m.Id == ms.MeasurementId WHERE m.PatientFirstName = ? Order By ModifiedDate DESC LIMIT 1', (imageID,))
    # Print the results
    if len(results) > 0:
        # Assuming the BLOB is in the second column, adjust the index if necessary
        image_data = results[0][0]        
        # Convert the binary data to an image
        image = Image.open(io.BytesIO(image_data))
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        return None

def get_frame_contours(frame):
    # Convert the frame to bytes
    success = cv2.imwrite('face.jpg', frame)
    if not success:
        raise Exception("Could not save face image.")
    
    relative_path = os.path.join(BasePath,'SparkSdkApi','SparkAI.API.exe')
    parameters = [os.path.join(BasePath,'face.jpg')]  # List of parameters

    # Combine the executable and parameters into a single command
    command = [relative_path] + parameters

    # Run the command
    result = subprocess.run(command, capture_output=True, text=True, cwd=os.path.join(BasePath,'SparkSdkApi'))
    if result.returncode != 0:
        raise Exception("Failed to extract frame contours.")

    leftContour, rightContour = parse_frame_contour(os.path.join(BasePath,"face_contour_points.json"))
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
ButtonFont = ('Helvetica 12 bold')
BorderWidth = 5
BackgroundColor = (193,0,0)
SparkImageSize = [2600, 1950]
ImageScaleFactor = (GetSystemMetrics(0) * 0.7)/MaxNumOfImages / SparkImageSize[1]
ImageSize = [int(SparkImageSize[0] * ImageScaleFactor), int(SparkImageSize[1] * ImageScaleFactor)]
ClearImage = set_clear_image()
IndicesQueue = []
Path = os.path.dirname(os.path.abspath(__file__))
BasePath = os.path.abspath(os.path.join(Path, os.pardir))
IconsPath = os.path.join(os.path.abspath(os.path.join(Path, os.pardir)), 'Icons')
state = State.SPLASH
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

# Start Monitoring the Spark4 DB
db_path = r'C:\ProgramData\Shamir\Spark4\DB\Spark4.db'
start_monitor_thread(db_path, my_callback)

### Main code ###
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

        if event == "-GET_IMAGE-":
            try:
                imageID = values["-IMAGE_ID-"]
                if imageID.strip() == "":
                    raise Exception("Please insert feasible image ID.")

                image = get_image(imageID)
                if image is None:
                    raise Exception("Image ID was not found.")

                # Process the frame with MediaPipe Face Detection
                results_detection = face_detection.process(image)
                leftContour, rightContour = get_frame_contours(image)
            except Exception as err:
                window["-OUTPUT-"].update(err)
                continue
                                                
            leftPupil = np.array([np.mean(leftContour[:,0]), np.mean(leftContour[:,1])])
            rightPupil = np.array([np.mean(rightContour[:,0]), np.mean(rightContour[:,1])])
            symmetryLine = np.array([[int((leftPupil[0]+rightPupil[0])/2)-1, int((leftPupil[1]+rightPupil[1])/2)+20], [int((leftPupil[0]+rightPupil[0])/2), int((leftPupil[1]+rightPupil[1])/2)-20]])
            lenspair = Lenspair(leftContour, rightContour, symmetryLine, leftPupil, rightPupil)

            ## Face detection
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results_mesh = face_mesh.process(rgb_image)

            if results_mesh.multi_face_landmarks:
                face_landmarks = results_mesh.multi_face_landmarks[0]
                faceFeatures = FaceFeatures(rgb_image, face_landmarks, image.shape)

            #x, y, w, h = detect_face_location(image)
            #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 4)
            #cv2.circle(image, (x, y), 10, (0, 255, 255), -1)
            
            if Debug:
                score = Score(faceFeatures, lenspair, 1)
                display_detailed_score(score)
                imageDebugMode = plot_features_debug_mode(image, lenspair, face_landmarks, faceFeatures)
                #faceDebugMode = centerize_face(imageDebugMode)

                cv2.putText(imageDebugMode, f'{score.total}', (100,100), cv2.FONT_HERSHEY_SIMPLEX, 4, (155, 0, 0), 5, cv2.LINE_AA)
                imgbytes = cv2.imencode('.png', imageDebugMode)[1].tobytes()                 # Convert the frame to a format that PySimpleGUI can display
                idx = get_index_for_display()
                set_image_for_display(imageDebugMode, idx)
            else:
                score = Score(faceFeatures, lenspair, 0)
                #face = centerize_face(image)

                cv2.putText(image, f'{score.total}', (100,100), cv2.FONT_HERSHEY_SIMPLEX, 4, (155, 0, 0), 5, cv2.LINE_AA)
                imgbytes = cv2.imencode('.png', image)[1].tobytes()                 # Convert the frame to a format that PySimpleGUI can display
                idx = get_index_for_display()
                set_image_for_display(image, idx)

            NumOfImages = min(NumOfImages + 1, MaxNumOfImages)

            for i in range(MaxNumOfImages):
                window[f'-GRAPH{i}-'].draw_image(data=Images4Display[f'{i}'], location=(0, 0))
            window["-OUTPUT-"].update('Image was loaded.')
            
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
            
window.close()
