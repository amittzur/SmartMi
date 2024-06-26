import numpy as np
import cv2
import PySimpleGUI as sg
import os.path
from win32api import GetSystemMetrics
from SmartMiObjects import State
import configparser
from DBMonitor import DBMonitor
from ScoringSystem import ScoringSystem


### Spark4 Monitor ###
def my_callback(image):
    global Images, AnnotatedImages, NumOfImages, MaxNumOfImages
    try:
        scs = ScoringSystem(image, BasePath)
        score = scs.calculate_score()
        annotatedImage = scs.get_annotated_image()

        cv2.putText(image, f'{score.total}', (100,100), cv2.FONT_HERSHEY_SIMPLEX, 4, (155, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(annotatedImage, f'{score.total}', (100,100), cv2.FONT_HERSHEY_SIMPLEX, 4, (155, 0, 0), 5, cv2.LINE_AA)
        idx = get_index_for_display()
        Images[f'{idx}'] = image
        AnnotatedImages[f'{idx}'] = annotatedImage

        if Debug:
            display_detailed_score(score)

        if DisplayAnnotations:
            set_image_for_display(annotatedImage, idx)
        else:
            set_image_for_display(image, idx)

    except Exception as err:
        window["-OUTPUT-"].update(err)
                                        
    NumOfImages = min(NumOfImages + 1, MaxNumOfImages)
    for i in range(MaxNumOfImages):
        window[f'-GRAPH{i}-'].draw_image(data=Images4Display[f'{i}'], location=(0, 0))
    #window["-OUTPUT-"].update('Image was loaded.')


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
        [sg.Button("Show\Hide annotations", size=(12, 2), font=ButtonFont, key="-SHOW_HIDE-")],
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

def set_clear_image():
    clearImage = np.full((ImageSize[0], ImageSize[1], 3), BackgroundColor[::-1], dtype=np.uint8)
    clearImage = cv2.copyMakeBorder(src=clearImage, top=BorderWidth, bottom=BorderWidth, left=BorderWidth, right=BorderWidth, value=BackgroundColor[::-1], borderType=cv2.BORDER_CONSTANT) 
    imgbytes = cv2.imencode('.png', clearImage)[1].tobytes()                 # Convert the images to a format that PySimpleGUI can display
    return imgbytes

def display_detailed_score(score):
    normalFont = 'Courier 9'
    titleFont = 'Courier 9 bold'
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
    sg.cprint('Frame area ratio = {}'.format(score.frameAreaRatio), font=normalFont)
    sg.cprint('Frame area score = {}'.format(score.frameArea), font=titleFont)
    sg.cprint('')
    sg.cprint('Total score = {}'.format(score.total), font=titleFont)

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
    Images4Display[f'{i}'] = imgbytes

def clear_images():
    global NumOfImages, IndicesQueue, Images, AnnotatedImages, Images4Display, window
    if NumOfImages == 0:
        return

    NumOfImages = 0
    IndicesQueue = []
    for i in range(MaxNumOfImages):
        Images[f'{i}'] = []
        AnnotatedImages[f'{i}'] = []
        Images4Display[f'{i}'] = []
        window[f'-GRAPH{i}-'].draw_image(data=ClearImage, location=(0, 0))

def read_config(file_path):
    # Initialize the ConfigParser
    config = configparser.ConfigParser()

    # Read the configuration file
    config.read(file_path)
    return config


### Global variables ###
config = read_config('Code/config.ini')
Images = {"0": [], "1": [], "2": []}
AnnotatedImages = {"0": [], "1": [], "2": []}
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
DisplayAnnotations = True

### Window layout ###
sg.theme("DarkRed1")
Shamir_Logo = os.path.join(IconsPath, 'ShamirNewLogo2.png')
StartIcon = cv2.imread(os.path.join(IconsPath, 'Start.png'))
window = create_splash_screen_layout_window()

# Start Monitoring the Spark4 DB
db_path = config['spark']['db_path'] #r'C:\ProgramData\Shamir\Spark4\DB\Spark4.db'

dm = DBMonitor(db_path)
dm.run(my_callback)

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
            
        if event == "-SHOW_HIDE-":
            DisplayAnnotations = not DisplayAnnotations
            for i in range(MaxNumOfImages):
                if len(Images[f'{i}']) > 0:
                    if DisplayAnnotations:
                        set_image_for_display(AnnotatedImages[f'{i}'], i)
                    else:
                        set_image_for_display(Images[f'{i}'], i)
                    window[f'-GRAPH{i}-'].draw_image(data=Images4Display[f'{i}'], location=(0, 0))

        if event == "-CLEAR_IMAGES-":
            ans = sg.popup('Are you sure you want to clear all images?', title='Clear image', background_color='lightgray', text_color='darkred',
                        custom_text=('Yes', 'No'), icon=os.path.join(IconsPath,'TitleLogo.ico'))
            if ans == 'Yes':
                clear_images()

        for i in range(MaxNumOfImages):
            if event == f'-GRAPH{i}-':
                if len(Images4Display[f'{i}']) > 0:
                    ans = sg.popup('Are you sure you want to clear the image?', title='Clear image', background_color='lightgray', text_color='darkred',
                                custom_text=('Yes', 'No'), icon=os.path.join(IconsPath,'TitleLogo.ico'))
                    if ans == 'Yes':
                        Images[f'{i}'] = []
                        AnnotatedImages[f'{i}'] = []
                        Images4Display[f'{i}'] = []
                        window[f'-GRAPH{i}-'].draw_image(data=ClearImage, location=(0, 0))
                        NumOfImages = NumOfImages - 1
                        IndicesQueue.remove(i)
                continue
            
window.close()