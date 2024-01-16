
# deactivate
# kivy_env_manual\Scripts\activate
# python first.py                 



import kivy

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
print("this is it hhhhhhhhhhhhhhhhhhhh")
print(sr.__file__)
import threading
import pymysql
from voice_auth_model import identify_speaker, preprocess, record_audio
from finger_auth_match import find_fingerprint_match

from kivy.uix.image import AsyncImage
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from threading import Event
from kivy.uix.screenmanager import Screen




        
class SECURERoomApp(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.authenticated_username = None
    def build(self):
        screen_manager = ScreenManager()
        

        screen_manager.add_widget(Builder.load_file("main.kv"))
        screen_manager.add_widget(Builder.load_file("face_authentification.kv"))
        screen_manager.add_widget(Builder.load_file("voice_authentification.kv"))
        screen_manager.add_widget(Builder.load_file("finger_authentification.kv"))
        screen_manager.add_widget(Builder.load_file("home.kv"))
    
    
        self.mydb = mysql.connector.connect(
           host="localhost",
           user="root", 
           password="04122002", 
           database="face_recognition"
        )
        c = self.mydb.cursor()
        c.execute("select name from face_recognition.users")
        for i in c.fetchall():
            print(i)
        return screen_manager
    
    
    
    
    
    
    def texture_from_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        buf = frame.tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
        texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        return Image(texture=texture)

    def show_face(self, threshold=0.9, duration=2):
         print("[INFO] loading model...")
         net = cv2.dnn.readNetFromCaffe("C:/Users/abdel/OneDrive/Bureau/detect_faces_video/Caffe_prototxt/deploy.prototxt", "C:/Users/abdel/OneDrive/Bureau/detect_faces_video/Caffe_model/Res10_300x300_SSD_iter_140000.caffemodel")

         print("[INFO] starting video stream...")
         vs = cv2.VideoCapture(0)
         time.sleep(1.0)

         face_detected_start = None

         while True:
                ret, frame = vs.read()
                if frame is None:
                   print("No frame captured. Check if the camera is connected and working properly.")
                   continue

                frame = imutils.resize(frame, width=500)
                (h, w) = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

                net.setInput(blob)
                detections = net.forward()

                face_detected = False

                for i in range(0, detections.shape[2]):
                  confidence = detections[0, 0, i, 2]
                  if confidence < threshold:
                     continue

                  face_detected = True
                  box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                  (startX, startY, endX, endY) = box.astype("int")

                  text = "{:.2f}%".format(confidence * 100)
                  y = startY - 10 if startY - 10 > 10 else startY + 10
                  cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                  cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

                if face_detected:
                   if face_detected_start is None:
                      face_detected_start = time.time()
                   elif time.time() - face_detected_start >= duration:
                       break
                else:
                    face_detected_start = None
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                   break

         cv2.destroyAllWindows()
         vs.release()


    def start_face_authentication(self):
              face_attempts = 0
              face_authenticated = False
              max_attempts = 3
              retry_delay = 10
              while face_attempts < 3:
            
            
                 self.show_face()
                 # Capture the user's face
                 cap = cv2.VideoCapture(0)
                 ret, frame = cap.read()
                 cap.release()

                  # Detect the face and compute its encoding
                 face_locations = face_recognition.face_locations(frame)
                 if len(face_locations) == 0:
                              print("No face detected. Please try again.")
                              warning_popup = Popup(title="No face detected", content=Label(text="No face detected. Please try again. "), size_hint=(None, None), size=(400, 200))
                              warning_popup.open()
                              face_attempts += 1
                              continue
                 face_encodings = face_recognition.face_encodings(frame, face_locations, num_jitters=10)
                 face_encoding = max(face_encodings, key=lambda encoding: face_recognition.face_distance([encoding], encoding)[0])

                 connection = pymysql.connect(
                        host="localhost", user="root", password="put here your password", database="face_recognition")
                 cursor = connection.cursor()
                 cursor.execute("SELECT name, username FROM users")
                 users = cursor.fetchall()
                 connection.close()

                 encodings_directory = "my_kivy_project\\encodings"
                # this local is the one where you save your faceencodings format   you can check opencv to know how to make photo for your face like that in the needded format

         # Compare the user's face with the stored encodings
                 face_authenticated = False
                 for name, username in users:
                    user_directory = os.path.join(encodings_directory, username)

                    if os.path.exists(user_directory):
                      for file in os.listdir(user_directory):
                         if file.endswith(".npy"):
                            encoding_file_path = os.path.join(user_directory, file)
                            stored_encoding = np.load(encoding_file_path)
                            match = face_recognition.compare_faces(
                            [stored_encoding], face_encoding)
                            if match[0]:
                               self.authenticated_username = username
                               self.root.current = 'voice_authentification' 
                               face_authenticated = True
                               print(f"Logged in as {name} ({username})")
                       
                               break

                    if face_authenticated:
                      break
               

        
                 if face_authenticated:
                    break
                 face_attempts += 1
                 if face_attempts == 3:
                    warning_popup = Popup(title="Face Authentication Failed", content=Label(text="Maximum attempts reached. You can retry after 5 minutes."), size_hint=(None, None), size=(400, 200))
                    warning_popup.open()
                    time.sleep(retry_delay)
                

   
   
   
   
   
   
   
   
   
   
   
# Add this function to preprocess the audio 
   
    def recognize_speech(self):

        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
             audio = recognizer.listen(source, timeout=10)

        try:
            recognized_text = recognizer.recognize_google(audio)
            print(f"Recognized: {recognized_text}")
        except sr.UnknownValueError:
            print("Could not understand audio")
            recognized_text = None
        except sr.RequestError as e:
            print(f"Error: {e}")
            recognized_text = None
    
        return recognized_text
    
   
    def authenticate_and_recognize_speech(self, bg_noise_dir=None):
        user_id, confidence = identify_speaker(bg_noise_dir)
        recognized_text = self.recognize_speech()

        return user_id, confidence, recognized_text
    
    
    def start_voice_authentification(self):
        # this  bg_noise_dir  is some noice voicec  so the model train  more perfectly you can add some noises as you needed and you can check kaggle   to more dataset
            bg_noise_dir = "background_noise"
            login_attempts = 0
            voice_authenticated=False
            while login_attempts < 3:

              is_authenticated, confidence ,recognized_text = self.authenticate_and_recognize_speech(  bg_noise_dir)
              print(recognized_text)
              print(is_authenticated)
              if is_authenticated== self.authenticated_username  and confidence > 0.7:
                  print(f"user{is_authenticated} loged in ")
                  # Get the expected_phrase from the database
                  connection = pymysql.connect(
                      host="localhost", user="root", password="put here your password", database="face_recognition")
                  cursor = connection.cursor()
                  cursor.execute("SELECT text_require FROM users WHERE username = %s", (is_authenticated))
                  result = cursor.fetchone()
                  connection.close()
                  if result:
                       expected_phrase = result[0]
                  else:
                       print("Could not retrieve the expected phrase from the database")
                       return
                  # Recognize the user's speech
                 
                  if recognized_text is not None and recognized_text.lower() == expected_phrase.lower():

                        self.root.current = 'finger_authentification' 
                        voice_authenticated = True
                        print(f"Logged in as {is_authenticated}    voice  knowing ")
                        break
                  else:
                      warning_popup = Popup(title="Voice Authentication Failed", content=Label(text="Voice Authentication Failed , your access word is wrong , pleas try again"), size_hint=(None, None), size=(400, 200))
                      warning_popup.open()
                      login_attempts += 1
                      continue
              else:
                  warning_popup = Popup(title="Voice Authentication Failed", content=Label(text="Voice Authentication Failed , voice didn recognized  , pleas try again"), size_hint=(None, None), size=(400, 200))
                  warning_popup.open()
                  login_attempts += 1
                  
              if voice_authenticated:
                   break
              
              
              if login_attempts == 3:
                    warning_popup = Popup(title="Voice Authentication Failed", content=Label(text="Voice Authentication Failed , voice didn recognized  , pleas try again"), size_hint=(None, None), size=(400, 200))
                    warning_popup.open()
                    return

 
    def start_finger_authentification(self):
        retry_delay = 10
        print("finger")
        login_attempts = 0
        path_f_chosen = "C:/Users/abdel/Downloads/SOCOFing/Altered/Altered-Hard/76__F_Right_thumb_finger_Obl.BMP"
        # this is the finger print pic that i chose to compare to the data and check if existe note that this pic is half erased to similate the finger scanner 
        # , also note that you can fet the fingers print data from kaggle chack read me to see the link  

        

        target_directory = "C:/Users/abdel/Downloads/SOCOFing/Real"
        def show_fingerprint_popup():
        # Create a BoxLayout to place the Image and Button
            layout = BoxLayout(orientation='vertical')

        # Show fingerprint image
            image = AsyncImage(source=path_f_chosen, size_hint=(1, 0.8))
            layout.add_widget(image)

        # Create a Button for closing the Popup
            ok_button = Button(text="OK")
            layout.add_widget(ok_button)

            finger_image_popup = Popup(title="Fingerprint", content=layout, size_hint=(None, None), size=(300, 400))

            finger_image_popup.open()

            ok_button.bind(on_press=finger_image_popup.dismiss)
            finger_image_popup.bind(on_dismiss=on_popup_dismiss)
            finger_image_popup.open()
       
        def on_popup_dismiss(*args):
          nonlocal login_attempts
           
          while login_attempts < 3:  
            connection = pymysql.connect(host="localhost", user="root", password="put here your password", database="face_recognition")
            cursor = connection.cursor()
            cursor.execute("SELECT finger_print FROM users")
            result = cursor.fetchall()
            connection.close()
  
            target_filenames = [record[0] for record in result]          
            match_found, score, matched_image = find_fingerprint_match(path_f_chosen,target_filenames)
            if match_found:
                 print(f"Match found: {matched_image}")
                 print(f"Score: {score}")
                 self.root.current = 'home' 
                      # Save the login time in the database
                 connection = pymysql.connect(
                     host="localhost", user="root", password="put here your password", database="face_recognition")
                 cursor = connection.cursor()
                 sql = "UPDATE users SET last_login=%s WHERE finger_print=%s"
                 cursor.execute(
                             sql, (time.strftime('%Y-%m-%d %H:%M:%S'), {matched_image}))
                 connection.commit()
                 connection.close()   
                 finger_authenticated=True                   
                 break
            else:
                 print("No match found.")
                 warning_popup = Popup(title="Finger Authentication Failed", content=Label(text="Please try again."), size_hint=(None, None), size=(400, 200))
                 warning_popup.open()
                 login_attempts += 1
                 continue
             
            if finger_authenticated:
                   break
              
            if login_attempts == 3:
                  QtWidgets.QMessageBox.warning(
                      self, "finger Authentication Failed", "Maximum attempts reached. Please try again.")
                  time.sleep(retry_delay)
                  return                  
        show_fingerprint_popup()    




if __name__ == '__main__':
    LabelBase.register(name="MPoppons",fn_regular="C:\\Users\\abdel\\OneDrive\\Bureau\\my_kivy_project\\fonts\\Poppins\\Poppins-Medium.ttf")
    LabelBase.register(name="BPoppons",fn_regular="C:\\Users\\abdel\\OneDrive\\Bureau\\my_kivy_project\\fonts\\Poppins\\Poppins-SemiBold.ttf")
   
    SECURERoomApp().run()
