import PySimpleGUI as sg
import os.path

Path = os.path.dirname(os.path.abspath(__file__))
IconsPath = os.path.join(os.path.abspath(os.path.join(Path, os.pardir)), 'Icons')
Shamir_Logo = os.path.join(IconsPath, 'ShamirNewLogo2.png')

layout1 = [
    # Your elements for layout 1
]

layout2 = [
    # Your elements for layout 2
]

layout3 = [
    # Your elements for layout 3
]

layout = [
    [sg.Text('A toggle button example')],
    [sg.Text('A graphical version'), sg.Button('', image_data=Shamir_Logo, key='-TOGGLE-GRAPHIC-', button_color=(sg.theme_background_color(), sg.theme_background_color()), border_width=0)],
    [sg.Button('On', size=(3, 1), button_color='white on green', key='-B-'), sg.Button('Exit')]
]
