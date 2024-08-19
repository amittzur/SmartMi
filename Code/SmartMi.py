import numpy as np
import cv2
import PySimpleGUI as sg
import os.path
import pandas as pd 
from win32api import GetSystemMetrics
from SmartMiObjects import Result, State
import configparser
from DBMonitor import DBMonitor
from ScoringSystem import ScoringSystem


### Spark4 Monitor ###
def my_callback(image):
    global Results, NumOfImages, MaxNumOfImages, ImageCaptured
    try:
        scs = ScoringSystem(image, BasePath)
        score = scs.calculate_score()
        annotatedImage = scs.get_annotated_image()

        cv2.putText(image, f'{score.total}', (100,100), cv2.FONT_HERSHEY_SIMPLEX, 4, (155, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(annotatedImage, f'{score.total}', (100,100), cv2.FONT_HERSHEY_SIMPLEX, 4, (155, 0, 0), 5, cv2.LINE_AA)
        idx = get_index_for_display()
        Results[f'{idx}'] = Result(image, annotatedImage, score, -1)

        if Debug:
            display_detailed_score(score)

        if DisplayAnnotations:
            set_image_for_display(annotatedImage, idx)
        else:
            set_image_for_display(image, idx)

        ImageCaptured = idx
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
    window['-IMAGE0_USER_SCORE-'].widget.configure(justify=sg.tk.CENTER)
    window['-IMAGE1_USER_SCORE-'].widget.configure(justify=sg.tk.CENTER)
    window['-IMAGE2_USER_SCORE-'].widget.configure(justify=sg.tk.CENTER)
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

    buttonsFrame = sg.Frame(layout=[
        [sg.Button("Show \ Hide annotations", size=(12, 2), font=ButtonFont, key="-SHOW_HIDE-")],
        [sg.Button("Clear images", size=(12, 2), font=ButtonFont, key="-CLEAR_IMAGES-")],
        [sg.Button("Close", size=(12, 2), font=ButtonFont, key="-CLOSE-")],
        [sg.VStretch()],
        [sg.Image(Shamir_Logo, subsample=2)]],
        title='', size=(350, 350), vertical_alignment='top', element_justification='center', border_width=0, pad=(80, 20))

    if Debug:
        leftLayout = [buttonsFrame, sg.Multiline(default_text='', size=(45,25), key="-OUTPUT_DEBUG-", reroute_cprint=True, no_scrollbar=True,
                            font='Helvetica 10', justification='c', text_color='white', background_color=backgroundColor, border_width=0, pad=((180,0), 20)),
                sg.Multiline(default_text='', size=(35,9), key="-OUTPUT-", no_scrollbar=True,
                            font='Courier 30 bold', justification='c', text_color='white', background_color=backgroundColor, border_width=0,pad=((30,0), 20))]
    else:
        leftLayout = [buttonsFrame, sg.Multiline(default_text='', size=(45,8), key="-OUTPUT-", no_scrollbar=True,
                            font='Courier 30 bold', justification='c', text_color='white', background_color=backgroundColor, border_width=1, pad=(40, 0))],

    layout = [leftLayout,
            [sg.Graph((ImageSize[1]+2*BorderWidth, ImageSize[0]+2*BorderWidth), (0, ImageSize[1]+2*BorderWidth), (ImageSize[0]+2*BorderWidth, 0), key='-GRAPH0-', enable_events=True, pad=((20,0), 0)),
            sg.Frame(layout=[[sg.Text('User score')], [sg.Listbox(["1","2","3","4","5","6","7","8","9","10"], size=(3, 10), key="-IMAGE0_USER_SCORE-", no_scrollbar=True, enable_events=True)],
                                [sg.Button("Update score", size=(6, 2), font=('Helvetica 8 bold'), key="-IMAGE0_USER_SCORE_UPDATE-")]],
                    title='', element_justification="center", size=(80,300), key="-IMAGE0_USER_FRAME-", border_width=0),
            sg.Graph((ImageSize[1]+2*BorderWidth, ImageSize[0]+2*BorderWidth), (0, ImageSize[1]+2*BorderWidth), (ImageSize[0]+2*BorderWidth, 0), key='-GRAPH1-', enable_events=True, pad=((120,0), 0)),
            sg.Frame(layout=[[sg.Text('User score')], [sg.Listbox(["1","2","3","4","5","6","7","8","9","10"], size=(3, 10), key="-IMAGE1_USER_SCORE-", no_scrollbar=True, enable_events=True)],
                                [sg.Button("Update score", size=(6, 2), font=('Helvetica 8 bold'), key="-IMAGE1_USER_SCORE_UPDATE-")]],
                    title='', element_justification="center", size=(80,300), key="-IMAGE1_USER_FRAME-", border_width=0),
            sg.Graph((ImageSize[1]+2*BorderWidth, ImageSize[0]+2*BorderWidth), (0, ImageSize[1]+2*BorderWidth), (ImageSize[0]+2*BorderWidth, 0), key='-GRAPH2-', enable_events=True, pad=((120,0), 0)),
            sg.Frame(layout=[[sg.Text('User score')], [sg.Listbox(["1","2","3","4","5","6","7","8","9","10"], size=(3, 10), key="-IMAGE2_USER_SCORE-", no_scrollbar=True, enable_events=True)],
                                [sg.Button("Update score", size=(6, 2), font=('Helvetica 8 bold'), key="-IMAGE0_USER_SCORE_UPDATE-")]],
                    title='', element_justification="center", size=(80,300), key="-IMAGE2_USER_FRAME-", border_width=0)]
        ]

    return layout

def set_clear_image():
    clearImage = np.full((ImageSize[0], ImageSize[1], 3), BackgroundColor[::-1], dtype=np.uint8)
    #clearImage = cv2.copyMakeBorder(src=clearImage, top=BorderWidth, bottom=BorderWidth, left=BorderWidth, right=BorderWidth, value=BackgroundColor[::-1], borderType=cv2.BORDER_CONSTANT) 
    clearImage = cv2.copyMakeBorder(src=clearImage, top=BorderWidth, bottom=BorderWidth, left=BorderWidth, right=BorderWidth, value=(255,255,255), borderType=cv2.BORDER_CONSTANT) 
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
    sg.cprint('Face circularity score = {}'.format(score.faceCircularity), font=normalFont)
    sg.cprint('Frame shape score = {}'.format(score.frameShape), font=titleFont)
    sg.cprint('')
    sg.cprint('DBL width ratio = {}'.format(score.DBLWidthRatio), font=normalFont)
    sg.cprint('DBL score = {}'.format(score.DBL), font=titleFont)
    sg.cprint('')
    sg.cprint('Frame color = {}'.format(score.lenspair.color), font=titleFont)
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
    global NumOfImages, IndicesQueue, Results, Images4Display, window
    if NumOfImages == 0:
        return

    NumOfImages = 0
    IndicesQueue = []
    for i in range(MaxNumOfImages):
        Results[f'{i}'] = []
        Images4Display[f'{i}'] = []
        window[f'-GRAPH{i}-'].draw_image(data=ClearImage, location=(0, 0))
        window[f'-IMAGE{i}_USER_SCORE-'].update(["1","2","3","4","5","6","7","8","9","10"])

def get_user_score(i):
    window["-OUTPUT-"].update(f'Please choose updated user score for image {i+1}.')
    while True:
        event, values = window.read()
        if event == f"-IMAGE{i}_USER_SCORE-":
            Results[f'{i}'].userScore = values[f"-IMAGE{i}_USER_SCORE-"][0]
            break
    return Results[f'{i}'].userScore

def save_result(i):
    global df_FrameFitting

    score = Results[f'{i}'].appScore
    new_row = {"TimeSignature": Results[f'{i}'].timeSignature, "FrameWidthRatio": score.frameWidthRatio, "FrameWidth": score.frameWidth,
                "LeftEyebrow": score.leftEyebrow, "RightEyebrow": score.rightEyebrow, "EyebrowsMatch": score.eyebrowsMatch,
                "LeftCheekLine": score.leftCheekLine, "RightCheekLine": score.rightCheekLine, "LowerCheekLine": score.lowerCheekLine,
                "CircularityPanelty": score.circularityPanelty, "FaceCircularity": score.faceCircularity, "FaceAspectRatio": score.faceAspectRatio, "FrameShape": score.frameShape,
                "DBLWidthRatio": score.DBLWidthRatio, "DBL": score.DBL, "Face_R": score.faceFeatures.skinColor[0], "Face_G": score.faceFeatures.skinColor[1], "Face_B": score.faceFeatures.skinColor[2],
                "Lenspair_R": score.lenspair.color[0], "Lenspair_G": score.lenspair.color[1], "Lenspair_B": score.lenspair.color[2], "ColorDiff": score.colorDiff, "FrameColor": score.frameColor,
                "FrameAreaRatio": score.frameAreaRatio, "FrameArea": score.frameArea,
                "W_frameWidth": score.W["frameWidth"], "W_eyebrowsMatch": score.W["eyebrowsMatch"], "W_lowerCheekLine": score.W["lowerCheekLine"],
                "W_frameShape": score.W["frameShape"], "W_DBL": score.W["DBL"], "W_frameColor": score.W["frameColor"], "W_frameArea": score.W["frameArea"],
                "Total": score.total, "UserScore": Results[f'{i}'].userScore
                }
    df_FrameFitting = df_FrameFitting._append(new_row, ignore_index=True)
    while True:
        try:
            df_FrameFitting.to_csv("FrameFitting.csv", index=False) 
            break
        except Exception as err:
            window["-OUTPUT-"].update("Could not save data to 'FrameFitting.csv'.\nPlease make sure the file is closed.")
            window.refresh()

    cv2.imwrite(os.path.join(BasePath, os.path.join("CapturedImages", f"{Results[f'{i}'].timeSignature}.jpg")), Results[f"{i}"].image)

def update_user_score(i):
    global df_FrameFitting

    Results[f'{i}'].userScore = values[f"-IMAGE{i}_USER_SCORE-"][0]
    df_FrameFitting.loc[df_FrameFitting['TimeSignature'] == Results[f'{i}'].timeSignature, 'UserScore'] = Results[f'{i}'].userScore

    while True:
        try:
            df_FrameFitting.to_csv("FrameFitting.csv", index=False) 
            break
        except Exception as err:
            window["-OUTPUT-"].update("Could not update 'FrameFitting.csv'.\nPlease make sure the file is closed.")
            window.refresh()

def read_config(file_path):
    # Initialize the ConfigParser
    config = configparser.ConfigParser()

    # Read the configuration file
    config.read(file_path)
    return config


### Global variables ###
config = read_config('Code/config.ini')
Results = {"0": [], "1": [], "2": []}
Images4Display = {"0": [], "1": [], "2": []}
MaxNumOfImages = len(Images4Display)
NumOfImages = 0
ButtonFont = ('Helvetica 12 bold')
BorderWidth = 3
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
Debug = False
DisplayAnnotations = True
ImageCaptured = -1


### Window layout ###
sg.theme("DarkRed1")
Shamir_Logo = os.path.join(IconsPath, 'ShamirNewLogo2.png')
StartIcon = cv2.imread(os.path.join(IconsPath, 'Start.png'))
window = create_splash_screen_layout_window()

# Start Monitoring the Spark4 DB
db_path = config['spark']['db_path'] #r'C:\ProgramData\Shamir\Spark4\DB\Spark4.db'

dm = DBMonitor(db_path)
dm.run(my_callback)

while True:
    try:
        df_FrameFitting = pd.read_csv("FrameFitting.csv") 
        break
    except Exception as err:
        window["-OUTPUT-"].update("Could not open 'FrameFitting.csv'.\nPlease make sure the file is closed.")


### Main code ###
while True:
    event, values = window.read(timeout=20)
    if event == "-EXIT-" or event == sg.WIN_CLOSED:
        break
    elif event == "-START-":
        window.close()
        window = create_app_layout_window()
        for i in range(3):
            window[f'-GRAPH{i}-'].draw_image(data=ClearImage, location=(0, 0))
        event, values = window.read(timeout=20)
        
    elif event == "-CLOSE-":
        clear_images()
        window.close()
        window = create_splash_screen_layout_window()
        event, values = window.read(timeout=20)

    if state != State.SPLASH:
        if ImageCaptured >= 0:
            get_user_score(ImageCaptured)
            save_result(ImageCaptured)
            window["-OUTPUT-"].update(f'Image {ImageCaptured+1} data was saved.')
            ImageCaptured = -1

        if event == "-SHOW_HIDE-":
            DisplayAnnotations = not DisplayAnnotations
            for i in range(MaxNumOfImages):
                if type(Results[f'{i}']) != list:
                    if DisplayAnnotations:
                        set_image_for_display(Results[f'{i}'].annotatedImage, i)
                    else:
                        set_image_for_display(Results[f'{i}'].image, i)
                    window[f'-GRAPH{i}-'].draw_image(data=Images4Display[f'{i}'], location=(0, 0))

        if event == "-CLEAR_IMAGES-":
            ans = sg.popup('Are you sure you want to clear all images?', title='Clear image', background_color='lightgray', text_color='darkred',
                        custom_text=('Yes', 'No'), icon=os.path.join(IconsPath,'TitleLogo.ico'))
            if ans == 'Yes':
                clear_images()

        if event == "-IMAGE0_USER_SCORE-":
            if type(Results['0']) != list:
                Results['0'].userScore = values["-IMAGE0_USER_SCORE-"][0]
            else:
                window['-IMAGE0_USER_SCORE-'].update(["1","2","3","4","5","6","7","8","9","10"])

        if event == "-IMAGE1_USER_SCORE-":
            if type(Results['1']) != list:
                Results['1'].userScore = values["-IMAGE1_USER_SCORE-"][0]
            else:
                window['-IMAGE1_USER_SCORE-'].update(["1","2","3","4","5","6","7","8","9","10"])

        if event == "-IMAGE2_USER_SCORE-":
            if type(Results['2']) != list:
                Results['2'].userScore = values["-IMAGE2_USER_SCORE-"][0]
            else:
                window['-IMAGE2_USER_SCORE-'].update(["1","2","3","4","5","6","7","8","9","10"])

        if event == "-IMAGE0_USER_SCORE_UPDATE-":
            if type(Results['0']) != list and len(values["-IMAGE0_USER_SCORE-"][0]) > 0:
                try:
                    update_user_score(0)
                    window["-OUTPUT-"].update('Image 1 score was updated.')
                except Exception as err:
                    window["-OUTPUT-"].update("Could not update the user score of image 1.")
                    
        if event == "-IMAGE1_USER_SCORE_UPDATE-":
            if type(Results['1']) != list and len(values["-IMAGE1_USER_SCORE-"][0]) > 0:
                try:
                    update_user_score(1)
                    window["-OUTPUT-"].update('Image 2 score was updated.')
                except Exception as err:
                    window["-OUTPUT-"].update("Could not update the user score of image 2.")

        if event == "-IMAGE2_USER_SCORE_UPDATE-":
            if type(Results['2']) != list and len(values["-IMAGE2_USER_SCORE-"][0]) > 0:
                try:
                    update_user_score(2)
                    window["-OUTPUT-"].update('Image 3 score was updated.')
                except Exception as err:
                    window["-OUTPUT-"].update("Could not update the user score of image 3.")

        for i in range(MaxNumOfImages):
            if event == f'-GRAPH{i}-':
                if len(Images4Display[f'{i}']) > 0:
                    ans = sg.popup('Are you sure you want to clear the image?', title='Clear image', background_color='lightgray', text_color='darkred',
                                custom_text=('Yes', 'No'), icon=os.path.join(IconsPath,'TitleLogo.ico'))
                    if ans == 'Yes':
                        Results[f'{i}'] = []
                        Images4Display[f'{i}'] = []
                        window[f'-GRAPH{i}-'].draw_image(data=ClearImage, location=(0, 0))
                        window[f'-IMAGE{i}_USER_SCORE-'].update(["1","2","3","4","5","6","7","8","9","10"])
                        NumOfImages = NumOfImages - 1
                        IndicesQueue.remove(i)
                continue

df_FrameFitting.dropna(subset=['TimeSignature'], inplace=True)
df_FrameFitting.to_csv("FrameFitting.csv", index=False) 

window.close()