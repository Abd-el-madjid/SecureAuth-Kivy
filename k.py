import kivy
from kivy.app import App
from kivy.uix.label import Label
import time
from kivy.core.text import LabelBase
from kivy.uix.screenmanager import ScreenManager
from kivymd.app import MDApp
from kivy.lang import Builder
from kivy.core.window import Window
import mysql.connector
import cv2
import time
import numpy as np
import face_recognition
import os
from pathlib import Path
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.uix.popup import Popup
import imutils
import speech_recognition as sr
import threading
import pymysql
from kivy.uix.image import AsyncImage
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from threading import Event
from kivy.uix.screenmanager import Screen
from kivy.uix.label import Label
from kivy.uix.floatlayout import FloatLayout

from kivy.graphics import Color, Rectangle

class SECURERoomApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.authenticated_username = None

    def update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def show_fingerprint_popup(self):
        # Path of the fingerprint image
        path_f_chosen = "C:/Users/abdel/Downloads/SOCOFing/Altered/Altered-Hard/76__F_Right_thumb_finger_Obl.BMP"

        # Create a FloatLayout to place the Image and Button
        layout = FloatLayout()

        # Set the background color of the layout to white
        with layout.canvas.before:
            Color(1, 1, 1, 1)  # colors range from 0-1 instead of 0-255
            self.rect = Rectangle(size=layout.size, pos=layout.pos)

        layout.bind(size=self.update_rect, pos=self.update_rect)

        # Show fingerprint image
        image = AsyncImage(source=path_f_chosen, size_hint=(1, 0.8), pos_hint={"top": 1}, allow_stretch=True, keep_ratio=False)
        layout.add_widget(image)

        # Create a Button for closing the Popup
        ok_button = Button(text="OK", size_hint=(.9, .1), pos_hint={"center_x": .5, "bottom": .1})
        layout.add_widget(ok_button)

        # Create the Popup
        finger_image_popup = Popup(title="Fingerprint", content=layout, size_hint=(None, None), size=(400, 500), auto_dismiss=False)

        # Bind the 'OK' button to the popup's dismiss event
        ok_button.bind(on_press=finger_image_popup.dismiss)

        # Open the popup
        finger_image_popup.open()

    def build(self):
        # Show the fingerprint popup when the application starts
        self.show_fingerprint_popup()


if __name__ == '__main__':
    SECURERoomApp().run()