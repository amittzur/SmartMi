from string import ascii_uppercase
import PySimpleGUI as sg

layout = [[sg.Listbox(ascii_uppercase, size=(3, 10), expand_x=True, key='LISTBOX')]]
window = sg.Window('Title', layout, finalize=True)
window['LISTBOX'].widget.configure(justify=sg.tk.CENTER)
window.read(close=True)
