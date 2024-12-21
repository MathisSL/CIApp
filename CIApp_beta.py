import tkinter as tk
from tkinter import Tk, Canvas, PhotoImage, filedialog, simpledialog, ttk, messagebox,Label, Frame, Button, Listbox, Scrollbar, Toplevel, Entry, messagebox
import customtkinter as ck
import cv2
import numpy as np
from PIL import Image, ImageTk
import time
import webbrowser
import pandas as pd
import os
import mediapipe as mp
import firebase_admin
from firebase_admin import credentials, auth, firestore, db
import requests
import uuid
import winsound
import math
import matplotlib.pyplot as plt
import datetime
import PosModuleHS as pm
import sys

print("Répertoire de travail actuel : ", os.getcwd())  # Affiche le répertoire courant

def ressource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller."""
    if hasattr(sys, '_MEIPASS'):
        print("Chemin _MEIPASS :", sys._MEIPASS)
        return os.path.join(sys._MEIPASS, relative_path)
    else:
        print("Chemin local :", os.path.abspath(relative_path))
        return os.path.abspath(relative_path)

json_path = ressource_path("ciapp-fbecc-firebase-adminsdk-z4yco-5597c6b972.json")
print("Chemin absolu JSON :", json_path)

if not os.path.exists(json_path):
    print("Erreur : fichier JSON introuvable.")
else:
    print("Fichier JSON trouvé :", json_path)

print("Chemin _MEIPASS :", getattr(sys, '_MEIPASS', 'Non défini'))


# Charger les informations d'authentification Firebase
cred = credentials.Certificate(json_path)
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://ciapp-fbecc-default-rtdb.europe-west1.firebasedatabase.app/'
})
firestore_db = firestore.client()
realtime_db = db.reference()

# Fonction pour écrire des données dans la Realtime Database
def write_data(path, data):
    ref = realtime_db.child(path.lstrip('/'))
    ref.set(data)

# Fonction pour obtenir ou créer un client ID unique
def get_or_create_client_id():
    """Get or create the client ID in Firestore."""
    doc_ref = firestore_db.collection(u'config').document(u'client_id')
    doc = doc_ref.get()

    if doc.exists:
        return doc.to_dict().get('client_id', None)
    else:
        client_id = str(uuid.uuid4())  # Génère un identifiant unique
        doc_ref.set({'client_id': client_id})  # Enregistre dans Firestore
        return client_id

# Configuration Google Analytics
MEASUREMENT_ID = "G-8YNJGKMLJT"
API_SECRET = "AIzaSyBq27wB72tQXN1B3oB_uenN-6Dj4A9ghZo"
MEASUREMENT_URL = "https://www.google-analytics.com/mp/collect"

# Fonction pour envoyer un événement à Google Analytics
def send_event(client_id, event_name, event_params=None):
    """
    Envoie un événement à Google Analytics.
    """
    if event_params is None:
        event_params = {}

    payload = {
        "client_id": client_id,
        "events": [
            {
                "name": event_name,
                "params": event_params
            }
        ]
    }

    url = f"{MEASUREMENT_URL}?measurement_id={MEASUREMENT_ID}&api_secret={API_SECRET}"

    try:
        print("Envoi des données à Google Analytics...")
        print("Payload :", payload)
        
        response = requests.post(url, json=payload)
        print(f"Réponse Google Analytics : {response.status_code} - {response.text}")

        if response.status_code == 204:
            print(f"Événement '{event_name}' envoyé avec succès !")
        else:
            print(f"Erreur lors de l'envoi de l'événement : {response.status_code} - {response.text}")

    except Exception as e:
        print(f"Erreur lors de l'envoi à Google Analytics : {e}")

# Fonction pour lire les événements dans Realtime Database et les envoyer à Google Analytics
def log_event_from_database():
    """
    Exemple de lecture des données depuis Realtime Database et envoi à Google Analytics.
    """
    try:
        snapshot = realtime_db.child('/events').get()  # Chemin
        if snapshot:
            for key, value in snapshot.items():
                # Vérifier si les données contiennent un utilisateur et un événement
                user_id = value.get('user_mail', None)
                event_name = value.get('event', None)
                if user_id and event_name:
                    send_event(user_id, event_name, value)
        else:
            print("Aucun événement trouvé dans la base de données.")
    except Exception as e:
        print(f"Erreur lors de la lecture des événements : {e}")

# Fonction pour créer un compte utilisateur
def sign_up(email, password):
    try:
        user = auth.create_user(email=email, password=password)
        messagebox.showinfo("Success", f"Account successfully created for {user.email} !")
        # Envoyer un événement "user_signup"
        send_event(user.email, "user_signup", {"method": "email"})
    except Exception as e:
        messagebox.showerror("Error", f"Impossible to create account try again : {e}")

# Fonction pour se connecter
def log_in(email, password, root):
    try:
        user = auth.get_user_by_email(email)
        messagebox.showinfo("Success", f"Connection successful for {user.email} !")
        send_event(user.email, "user_login", {"method": "email", "debug_mode": True})
        write_data('/events/user_login', {'user_mail': user.email, 'event': 'user_login'})     
        setup_ui2(root)
    except Exception as e:
        messagebox.showerror("Error", f"Impossible to connect : {e}")


# Lien vers le google form

def open_link():
    webbrowser.open_new("https://forms.gle/Y4SmkSCWsnotdiBm6")

def open_link2():
    webbrowser.open_new("https://www.freeconvert.com/fr/mov-to-mp4")

def open_link3():
    webbrowser.open_new("https://youtu.be/BboGB5kvgAM")

def open_link4():
    webbrowser.open_new("https://ivcam.fr.download.it/")

# Scipts des exercices
# Script Anatole ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def code_analyse_posturale():
    # Initialize MediaPipe pose and drawing utilities
    global video_path
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Initialize global variables
    video_path = ""
    angle_data = []

    def save_to_excel(): # VF
        """Save the series data to an Excel file."""  
        df = pd.DataFrame(angle_data, columns=['Time', 'Angle shoulder elbow wrist left', 'Angle shoulder elbow wrist wrist', 'Angle_bassin_genou_cheville_gauche', 'Angle_bassin_genou_cheville_droite', 'Angle_cheville_bassin_epaule_gauche', 'Angle_cheville_bassin_epaule_droit', 'Angle_epaule_oreille_nez_gauche', 'Angle_epaule_oreille_nez_droit', 'Angle_cheville_gauche', 'Angle_cheville_droite', 'Angle_poignet_epaule_bassin_gauche', 'Angle_poignet_epaule_bassin_droit', 'pos_hand_left_x', 'pos_hand_left_y', 'pos_hand_right_x', 'pos_hand_right_y', 'pos_shoulder_left_x', 'pos_shoulder_left_y', 'pos_shoulder_right_x', 'pos_shoulder_right_y', 'pos_foot_left_x', 'pos_foot_left_y',' pos_foot_right_x',' pos_foot_right_y', 'pos_hip_left_x', 'pos_hip_left_y', 'pos_hip_right_x', 'pos_hip_right_y', 'pos_knee_left_x', 'pos_knee_left_y', 'pos_knee_right_x', 'pos_knee_right_y', 'pos_elbow_left_x', 'pos_elbow_left_y', 'pos_elbow_right_x',' pos_elbow_right_y', 'pos_nose_x', 'pos_nose_y'])
        file_path = 'output.xlsx'
        try:
            df.to_excel(file_path, index=False)
            print(f"DataFrame saved to {file_path}")
            df.to_csv('repetitions.csv', index=False)
            os.system(f'start excel "{file_path}"')
        except PermissionError:
            print(f"Permission denied: Unable to save {file_path}. Please close the file if it is open and try again.")

    def setup_ui():
        """Setup the main application window and UI elements."""
        global table
        window = tk.Toplevel()
        window.geometry("1280x720")
        window.title("Ultimate movement Analysis")
        ck.set_appearance_mode("dark")

        # UI Elements

        saveButton = ck.CTkButton(window, text='SAVE TO EXCEL', command=save_to_excel, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="green")
        saveButton.place(x=575, y=500)

        # Button to select video
        videoButton = ck.CTkButton(window, text='Select Video', command=select_video, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
        videoButton.place(x=600, y=620)

        # Button to process video
        processButton = ck.CTkButton(window, text='Process Video', command=lambda: process_video(lmain), height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
        processButton.place(x=750, y=620)

        btn_signaler = ck.CTkButton(window, text="Convert your video .mov in .mp4 for analysis", command=open_link2, height=40, width=200, text_color="white", fg_color="red")
        btn_signaler.place(x=200, y=620)

        info_label = ck.CTkLabel(window, text="Processed video saved as output.avi", font=("Arial", 10), text_color="black")
        info_label.place(x=10, y=60)

        frame = tk.Frame(window, height=480, width=480)
        frame.place(x=10, y=90)
        lmain = tk.Label(frame)
        lmain.place(x=0, y=0)


        # Progress Bar
        progress_bar = ttk.Progressbar(window, orient='horizontal', length=400, mode='determinate')
        progress_bar.place(x=480, y=300)


        return window, lmain, progress_bar

    def select_video():
        """Select a video file."""
        global video_path
        video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])

    def calculate_angle(a, b, c):
        """Calculate the angle between three points."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return round(angle, 2)
        


    def process_video(lmain):
        """Process the selected video and save the repetitions."""
        global video_path, progress_bar
        if not video_path:
            return

        cap = cv2.VideoCapture(video_path)
        start_time = time.time()
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            for frame_num in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                try:
                    landmarks = results.pose_landmarks.landmark
                    shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    left_hand = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    right_hand = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    left_mouth = [landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].x, landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].y]
                    right_mouth = [landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].x, landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].y]
                    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x, landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
                    right_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]
                    nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]
                    left_heal = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
                    right_heal = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]
                    left_foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
                    right_foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
                    

                    # Calculer les angles
                    angle_epaule_coude_poignet_gauche = calculate_angle(shoulder_left, left_elbow, left_wrist)
                    angle_epaule_coude_poignet_droit = calculate_angle(shoulder_right, right_elbow, right_wrist)
                    angle_bassin_genou_cheville_gauche = calculate_angle(left_hip, left_knee, left_ankle)
                    angle_bassin_genou_cheville_droite = calculate_angle(right_hip, right_knee, right_ankle)
                    angle_cheville_bassin_epaule_gauche = calculate_angle(left_ankle, left_hip, shoulder_left)
                    angle_cheville_bassin_epaule_droit = calculate_angle(right_ankle, right_hip, shoulder_right)
                    angle_epaule_oreille_nez_gauche = calculate_angle(shoulder_left, left_ear, nose)
                    angle_epaule_oreille_nez_droit = calculate_angle(shoulder_right, right_ear, nose)
                    angle_cheville_gauche = calculate_angle(left_knee, left_heal, left_foot_index)
                    angle_cheville_droite = calculate_angle(right_knee, right_heal, right_foot_index)
                    angle_poignet_epaule_bassin_gauche = calculate_angle(left_wrist, shoulder_left, left_hip)
                    angle_poignet_epaule_bassin_droit = calculate_angle(right_wrist, shoulder_right, right_hip)

                    # Positions d'éléments du corps
                    pos_hand_left_x = left_wrist[0]
                    pos_hand_left_y = 1-left_wrist[1]
                    pos_hand_right_x = right_wrist[0]
                    pos_hand_right_y = 1-right_wrist[1]
                    pos_shoulder_left_x = shoulder_left[0]
                    pos_shoulder_left_y = 1-shoulder_left[1]
                    pos_shoulder_right_x = shoulder_right[0]
                    pos_shoulder_right_y = 1-shoulder_right[1]
                    pos_foot_left_x = left_ankle[0]
                    pos_foot_left_y = 1-left_ankle[1]
                    pos_foot_right_x = right_ankle[0]
                    pos_foot_right_y = 1-right_ankle[1]
                    pos_hip_left_x = left_hip[0]
                    pos_hip_left_y = 1-left_hip[1]
                    pos_hip_right_x = right_hip[0]
                    pos_hip_right_y = 1-right_hip[1]
                    pos_knee_left_x = left_knee[0]
                    pos_knee_left_y = 1-left_knee[1]
                    pos_knee_right_x = right_knee[0]
                    pos_knee_right_y = 1-right_knee[1]
                    pos_elbow_left_x = left_elbow[0]
                    pos_elbow_left_y = 1-left_elbow[1]
                    pos_elbow_right_x = right_elbow[0]
                    pos_elbow_right_y = 1-right_elbow[1]
                    pos_nose_x = nose[0]
                    pos_nose_y = 1-nose[1]

                    # Calculer le temps écoulé
                    elapsed_time = time.time() - start_time
                    # Enregistrer l'angle et le timestamp
                    angle_data.append((elapsed_time, angle_epaule_coude_poignet_gauche, angle_epaule_coude_poignet_droit, angle_bassin_genou_cheville_gauche, angle_bassin_genou_cheville_droite, angle_cheville_bassin_epaule_gauche, angle_cheville_bassin_epaule_droit, angle_epaule_oreille_nez_gauche, angle_epaule_oreille_nez_droit, angle_cheville_gauche, angle_cheville_droite, angle_poignet_epaule_bassin_gauche, angle_poignet_epaule_bassin_droit,pos_hand_left_x, pos_hand_left_y, pos_hand_right_x, pos_hand_right_y, pos_shoulder_left_x, pos_shoulder_left_y, pos_shoulder_right_x, pos_shoulder_right_y, pos_foot_left_x, pos_foot_left_y, pos_foot_right_x, pos_foot_right_y, pos_hip_left_x, pos_hip_left_y, pos_hip_right_x, pos_hip_right_y, pos_knee_left_x, pos_knee_left_y, pos_knee_right_x, pos_knee_right_y, pos_elbow_left_x, pos_elbow_left_y, pos_elbow_right_x, pos_elbow_right_y, pos_nose_x, pos_nose_y))
                    print(angle_data)

                except:
                    pass

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), 
                                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))               

                out.write(image)


            # Redimensionner l'image pour qu'elle corresponde au cadre tout en gardant les proportions
            frame_width, frame_height = 480, 480 # Taille définie pour l'affichage
            height, width, _ = image.shape
            scale = min(frame_width / width, frame_height / height)
            resized_width, resized_height = int(width * scale), int(height * scale)

            image_resized = cv2.resize(image, (resized_width, resized_height))
            img = Image.fromarray(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            lmain.imgtk = imgtk
            lmain.configure(image=imgtk)
            lmain.update()

            # Update the progress bar
            progress = (frame_num / total_frames) * 100
            progress_bar['value'] = progress
            progress_bar.update_idletasks()  # Force the UI to update

    

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def main():
        global lmain, progress_bar
        """Main function to run the pose detection and UI update loop."""
        window, lmain, progress_bar = setup_ui()
        window.mainloop()

    if __name__ == "__main__":
        main()

def code_enhanced_HS():
    # Créer une nouvelle fenêtre Toplevel
    hs_window = tk.Toplevel()
    hs_window.title("Handstand Project Interface")
    hs_window.geometry("1200x800") 
    cap = cv2.VideoCapture(camera_index)

    # Obtenir les propriétés de la vidéo
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Définir la largeur des bandes noires latérales
    side_bar_width = 200 
    new_frame_width = frame_width + 2 * side_bar_width

    # Initialize the PoseDetector
    detector = pm.PoseDetector()

    # Variables
    global record_time, timer_started, start_time, elapsed_time, average_values, max_average, mirror_video, time_goal, recording, pTime
    pTime = 0
    timer_started = False
    start_time = 0
    elapsed_time = 0
    record_time = 0
    average_values = []
    max_average = 0
    mirror_video = False
    time_goal = None  # Variable to store the time goal

    # Video recording variables
    recording = False
    out = None

    # Load the record time from file
    try:
        with open('record_time.txt', 'r') as file:
            record_time = float(file.read())
    except:
        record_time = 0

    # Function to reset the PR
    def reset_pr():
        global record_time
        record_time = 0
        with open('record_time.txt', 'w') as file:
            file.write(str(record_time))
        pr_label.config(text=f'PR : {record_time:.2f}s')

    def reset_time_history():
        for file in os.listdir('time_history'):
            if file.startswith('times_') and file.endswith('.txt'):
                os.remove(os.path.join('time_history', file))
        time_history_listbox.delete(0, tk.END)

    # Function to save and plot average values
    def save_and_plot_average_values():
        global average_values
        with open('average_values.txt', 'w') as file:
            for value in average_values:
                file.write(f"{value}\n")
        plt.plot(average_values)
        plt.xlabel('Time')
        plt.ylabel('Average Angle')
        plt.title('Average Angle Over Time')
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'Graph_perf/average_angle_plot{timestamp}.png')
        plt.show()

    # Function to toggle mirroring
    def toggle_mirror():
        global mirror_video
        mirror_video = not mirror_video

    # Function to load time history
    def load_time_history():
        time_history_listbox.delete(0, tk.END)
        try:
            files = sorted(os.listdir('time_history'))
            for file in files:
                if file.startswith('times_') and file.endswith('.txt'):
                    with open(os.path.join('time_history', file), 'r') as f:
                        time_entry = f.read().strip()
                        time_history_listbox.insert(tk.END, time_entry)
        except FileNotFoundError:
            pass

    # Function to set the time goal
    def set_time_goal():
        def save_time_goal():
            global time_goal
            try:
                time_goal = int(goal_entry.get())
                goal_window.destroy()
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter a valid integer.")

        goal_window = Toplevel(hs_window)
        goal_window.title("Set Time Goal")
        goal_window.geometry("300x100")

        goal_label = Label(goal_window, text="Enter Time Goal (seconds):")
        goal_label.pack(pady=10)

        goal_entry = Entry(goal_window)
        goal_entry.pack(pady=5)

        save_button = Button(goal_window, text="Save", command=save_time_goal)
        save_button.pack(pady=5)

    # Function to show a non-blocking alert
    def show_non_blocking_alert(message):
        winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS | winsound.SND_LOOP | winsound.SND_ASYNC)
        alert_window = Toplevel(hs_window) ## Top level deja pris
        alert_window.title("Alert")
        alert_window.geometry("300x100")

        alert_label = Label(alert_window, text=message)
        alert_label.pack(pady=20)

        def stop_sound():
            winsound.PlaySound(None, winsound.SND_ASYNC)  # Arrêter le son
            alert_window.destroy()

        ok_button = Button(alert_window, text="OK", command=stop_sound)
        ok_button.pack(pady=5)

    def update_frame():
        global pTime,timer_started, start_time, elapsed_time, record_time, max_average, imgtk
        global recording, out, time_goal

        processing_start_time = time.time()

        ret, frame = cap.read()
        if ret:
            if mirror_video:
                frame = cv2.flip(frame, 1)

            # Créer une nouvelle image avec des bandes grises sur les côtés
            new_frame = np.full((frame_height, new_frame_width, 3), 230, dtype=np.uint8)

            # Placer la frame d'origine au centre de la nouvelle image
            new_frame[:, side_bar_width:side_bar_width+frame_width] = frame

            # Traitement de la frame (détection de pose)
            new_frame = detector.findPose(new_frame)
            lmList = detector.findPosition(new_frame, draw=True)

            if len(lmList) != 0:
                x1, y_epaule = lmList[12][1:]
                x2, y_bassin = lmList[24][1:]
                x3, y_genou = lmList[26][1:]
                # Calcul des angles et des pourcentages
                angle_genou, diff_genou, color_genou = detector.findAngle(new_frame, 28, 26, 24)
                angle_bassin, diff_bassin, color_bassin = detector.findAngle(new_frame, 26, 24, 12)
                angle_epaule, diff_epaule, color_epaule = detector.findAngle(new_frame, 14, 12, 24)
                moyenne_angles = (angle_genou + angle_bassin + angle_epaule) / 3
                per_moyenne = int(np.interp(moyenne_angles, (120, 180), (0, 100)))

                # Vérifier si les trois angles sont dans le vert
                if diff_bassin <= 45 and diff_epaule <= 45 and diff_genou <= 45 and y_epaule > y_bassin > y_genou:
                    if not recording:
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        video_path = f'HS_videos/HS_record_{timestamp}.avi'
                        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (new_frame_width, frame_height))
                        recording = True
                    if not timer_started:
                        timer_started = True
                        start_time = time.time()
                    else:
                        elapsed_time = time.time() - start_time
                    # Enregistrer les valeurs de moyenne
                    average_values.append(moyenne_angles)
                    # Mettre à jour la valeur maximale de la moyenne
                    if moyenne_angles > max_average:
                        max_average = moyenne_angles
                else:
                    if recording:
                        recording = False
                        out.release()
                    if timer_started:
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        if elapsed_time >= 1:  # Only save if elapsed time is ... second or more
                            date_str = datetime.datetime.now().strftime("%d_%m-%H-%M-%S")
                            with open(f'time_history/times_{date_str}.txt', 'w') as file:
                                file.write(f"{elapsed_time:.2f}s")
                            load_time_history()   
                        timer_started = False
                        elapsed_time = elapsed_time

                if elapsed_time > record_time:
                    record_time = elapsed_time
                    with open('record_time.txt', 'w') as file:
                        file.write(str(record_time))
                    pr_label.config(text=f'PR : {record_time:.2f}s')            

                # Check if elapsed_time reaches the time_goal
                if time_goal is not None and elapsed_time >= time_goal:
                    show_non_blocking_alert(f"You have reached your time goal of {time_goal} seconds!")
                    time_goal = None  # Reset the time goal after reaching it

                # Conversion des angles en pourcentages
                per_genou = int(np.interp(angle_genou, (120, 180), (0, 100)))
                per_bassin = int(np.interp(angle_bassin, (120, 180), (0, 100)))
                per_epaule = int(np.interp(angle_epaule, (120, 180), (0, 100)))

                # Positions des informations sur les côtés ajustées en fonction de new_frame
                left_x = int(new_frame_width * 0.05) - 40
                right_x = new_frame_width - side_bar_width + 50
                y_start = int(frame_height * 0.1) + 60
                y_offset = 150
                bar_height = 100

                # Dessiner les barres de progression
                # Genou
                cv2.rectangle(new_frame, (right_x , y_start), (right_x + 50, y_start - per_genou), color_genou, -1)
                cv2.rectangle(new_frame, (right_x , y_start - bar_height), (right_x + 50, y_start), (0, 0, 0), 1)
                cv2.putText(new_frame, f'Genou: {per_genou}%', (right_x, y_start + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_genou, 2)

                # Bassin
                cv2.rectangle(new_frame, (right_x , y_start + y_offset), (right_x + 50, y_start + y_offset - per_bassin), color_bassin, -1)
                cv2.rectangle(new_frame, (right_x , y_start + y_offset - bar_height), (right_x + 50, y_start + y_offset), (0, 0, 0), 1)
                cv2.putText(new_frame, f'Bassin: {per_bassin}%', (right_x, y_start + y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bassin, 2)

                # Épaule
                cv2.rectangle(new_frame, (right_x, y_start + 2*y_offset), (right_x + 50, y_start + 2*y_offset - per_epaule), color_epaule, -1)
                cv2.rectangle(new_frame, (right_x, y_start + 2*y_offset - bar_height), (right_x + 50, y_start + 2*y_offset), (0, 0, 0), 1)
                cv2.putText(new_frame, f'Epaule: {per_epaule}%', (right_x, y_start + 2*y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_epaule, 2)
                
                # Moyenne
                if per_moyenne >= 90:
                    color_moyenne = (255, 0, 255)
                elif 90 > per_moyenne >= 75:
                    color_moyenne = (0, 255, 0)
                elif 75 > per_moyenne >= 65:
                    color_moyenne = (0, 165, 255)
                else:
                    color_moyenne = (0, 0, 255)
                cv2.rectangle(new_frame, (left_x, y_start + 2*y_offset - 20), (left_x + 50, y_start + 2*y_offset - per_moyenne*2 - 20), color_moyenne, -1)
                cv2.rectangle(new_frame, (left_x, y_start + 2*y_offset - 200 - 20), (left_x + 50, y_start + 2*y_offset - 20), (255, 255, 255), 1)
                cv2.putText(new_frame, f'Moyenne: {per_moyenne}%', (left_x, y_start + 2*y_offset +5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_moyenne, 2)

                # Afficher la moyenne des trois angles
                cv2.putText(new_frame, f'Avg Angle: {moyenne_angles:.2f}', (left_x, y_start + 2*y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

                # Afficher le timer 
                cv2.putText(new_frame, f'Timer: {elapsed_time:.2f}s', (left_x, y_start + y_offset//5 -10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                # Afficher le record 
                cv2.putText(new_frame, f'Record: {record_time:.2f}s', (left_x, y_start + 2*y_offset//5 -10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                # Afficher les FPS
                cTime = time.time()
                fps_display = 1 / (cTime - pTime + 1e-8)
                pTime = cTime
                cv2.putText(new_frame, f'{int(fps_display)} FPS', (left_x, y_start -10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Enregistrer la frame si enregistrement actif
                if recording and out is not None:
                    out.write(new_frame)

            img_rgb = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)

            # Calculate processing time
            processing_end_time = time.time()
            processing_time = processing_end_time - processing_start_time
            processing_time_label.config(text=f'Processing Time: {processing_time:.2f}s')

            timer_label.config(text=f'Timer : {elapsed_time:.2f}s') 

            # Planification de la prochaine frame
            video_label.after(1, update_frame)

        else:
            # Si la vidéo est terminée, réinitialiser à la première frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            video_label.after(1, update_frame)

    # Interface utilisateur dans la nouvelle fenêtre
    top_panel = Frame(hs_window)
    top_panel.pack(side=tk.TOP, padx=10, pady=10)

    timer_label = Label(top_panel, text="Timer : 0.00s", font=("Helvetica", 16))
    timer_label.pack(side=tk.LEFT, padx=20)

    pr_label = Label(top_panel, text=f'PR : {record_time:.2f}s', font=("Helvetica", 16))
    pr_label.pack(side=tk.LEFT, padx=20)

    set_goal_button = Button(top_panel, text="Set Goal", command=set_time_goal, font=("Helvetica", 16))
    set_goal_button.pack(side=tk.LEFT, padx=20)

    reset_button = Button(top_panel, text="Reset PR", command=reset_pr, font=("Helvetica", 16))
    reset_button.pack(side=tk.LEFT, padx=20)

    reset_time_history_button = Button(top_panel, text="Reset Time History", command=reset_time_history, font=("Helvetica", 16))
    reset_time_history_button.pack(side=tk.LEFT, padx=20)

    plot_button = Button(top_panel, text="Average Values", command=save_and_plot_average_values, font=("Helvetica", 16))
    plot_button.pack(side=tk.LEFT, padx=20)

    mirror_button = Button(top_panel, text="Mirror Video", command=toggle_mirror, font=("Helvetica", 16))
    mirror_button.pack(side=tk.LEFT, padx=20)

    processing_time_label = Label(top_panel, text="Processing Time: 0.00s", font=("Helvetica", 16))
    processing_time_label.pack(side=tk.LEFT, padx=30)

    middle_panel = Frame(hs_window)
    middle_panel.pack(side=tk.LEFT, padx=10, pady=10)

    video_label = Label(middle_panel)
    video_label.pack()

    right_panel = Frame(hs_window)
    right_panel.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)

    time_history_label = Label(right_panel, text="Time History", font=("Helvetica", 16))
    time_history_label.pack(pady=5)

    scrollbar = Scrollbar(right_panel)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    time_history_listbox = Listbox(right_panel, font=("Helvetica", 14), yscrollcommand=scrollbar.set)
    time_history_listbox.pack(fill=tk.BOTH, expand=True)

    scrollbar.config(command=time_history_listbox.yview)
    load_time_history()

    update_frame()
    hs_window.mainloop()

    if out is not None:
        out.release()
    cap.release()
    cv2.destroyAllWindows()
# Fin de script Anatole--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def code_upload_planche():
    # Initialisation de mediapipe
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Initialisationn des variables globales
    global counter, stage, PR_rep, series_data, current_series, timer_started, start_time
    counter = 0
    stage = None
    PR_rep = 0
    series_data = []  # Liste pour stocker les séries
    current_series = 1
    timer_started = False
    start_time = 0
    video_path = ""
    angle_data = []

    def save_to_excel():
        """Sauver les données des séries dans un fichier excel."""  
        df = pd.DataFrame(angle_data, columns=['Time', 'left_knee_foot_hip_angle','left_knee_hip_shoulder_angle','right_knee_hip_shoulder_angle2','right_knee_foot_hip_angle2','left_shoulder_elbow_wrist_angle','right_shoulder_elbow_wrist_angle2'])
        file_path = 'output.xlsx'
        try:
            df.to_excel(file_path, index=False)
            print(f"DataFrame saved to {file_path}")
            df.to_csv('repetitions.csv', index=False)
            os.system(f'start excel "{file_path}"')
        except PermissionError:
            print(f"Permission denied: Unable to save {file_path}. Please close the file if it is open and try again.")

    def setup_ui():
        """Setup the main application window and UI elements."""
        global table
        window = tk.Toplevel()
        window.geometry("1280x720")
        window.title("Planche timer upload video")
        ck.set_appearance_mode("dark")

        # UI Elements
        classLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10, text='Statistics : ')
        classLabel.place(x=10, y=41)

        counterLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10, text='Planche hold')
        counterLabel.place(x=160, y=1)

        PRLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10, text='PR planche')
        PRLabel.place(x=160, y=580)

        counterBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="#ff7b00", text='0')
        counterBox.place(x=160, y=41)

        PRBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="Green", text='0')
        PRBox.place(x=160, y=620)

        resetButton = ck.CTkButton(window, text='RESET', command=reset_counter, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="red")
        resetButton.place(x=10, y=620)

        prResetButton = ck.CTkButton(window, text='RESET PR', command=lambda: reset_pr_time(PRBox), height=40, width=120, font=("Arial", 20), text_color="white", fg_color="red")
        prResetButton.place(x=300, y=620)

        saveButton = ck.CTkButton(window, text='SAVE TO EXCEL', command=save_to_excel, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="green")
        saveButton.place(x=575, y=500)

        btn_signaler = ck.CTkButton(window, text="Convert .mov in .mp4 for analysis", command=open_link2, height=40, width=200, text_color="white", fg_color="red")
        btn_signaler.place(x=575, y=20)

        # Button to select video
        videoButton = ck.CTkButton(window, text='Select Video', command=select_video, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
        videoButton.place(x=600, y=620)

        # Button to process video
        processButton = ck.CTkButton(window, text='Process Video', command=lambda: process_video(counterBox, PRBox, lmain), height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
        processButton.place(x=750, y=620)

        frame = tk.Frame(window, height=480, width=480)
        frame.place(x=10, y=90)
        lmain = tk.Label(frame)
        lmain.place(x=0, y=0)


        # Progress Bar
        progress_bar = ttk.Progressbar(window, orient='horizontal', length=400, mode='determinate')
        progress_bar.place(x=480, y=300)

        return window, lmain, counterBox, PRBox, progress_bar

    def select_video():
        """Select a video file."""
        global video_path
        video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])

    def reset_counter(): 
        """Reset the counter and timer to zero."""
        global counter
        counter = 0 

    def reset_pr_time(PRBox):
        """Reset the PR rep and update the display."""
        global PR_rep
        PR_rep = 0
        PRBox.configure(text='0')

    def calculate_angle(a, b, c):
        """Calculate the angle between three points."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return round(angle, 2)

    def process_video(counterBox,PRBox, lmain):
        """Process the selected video and save the repetitions."""
        global video_path, counter, PR_rep, series_data, current_series, progress_bar
        if not video_path:
            return

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            for frame_num in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                try:
                    landmarks = results.pose_landmarks.landmark
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    left_hand = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    right_hand = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    left_mouth = [landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].x, landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].y]
                    right_mouth = [landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].x, landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].y]
                    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]


                    knee_foot_hip_angle = calculate_angle(left_ankle, left_knee, left_hip)
                    knee_hip_shoulder_angle = calculate_angle(left_knee, left_hip, left_shoulder)
                    knee_hip_shoulder_angle2 = calculate_angle(right_knee, right_hip, right_shoulder)
                    knee_foot_hip_angle2 = calculate_angle(right_ankle, right_knee, right_hip)
                    shoulder_elbow_wrist_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                    shoulder_elbow_wrist_angle2 = calculate_angle(right_shoulder, right_elbow, right_wrist)

                    # Enregistrer l'angle et le timestamp
                    angle_data.append((time.time(), knee_foot_hip_angle,knee_hip_shoulder_angle,knee_hip_shoulder_angle2,knee_foot_hip_angle2,shoulder_elbow_wrist_angle,shoulder_elbow_wrist_angle2))
                    print(angle_data)

    ########################################################################################################
    ########################################################################################################
                                        # Section conditions pour compteur
    ########################################################################################################
    ########################################################################################################                                
                    # Condition pour démarrer le chrono pour la planche
                    global timer_started, start_time, PR_rep
                    if (left_wrist[1] > left_elbow[1] > left_shoulder[1] or right_wrist[1] > right_elbow[1] > right_shoulder[1]) and knee_hip_shoulder_angle > 155 and knee_foot_hip_angle > 155 and (shoulder_elbow_wrist_angle > 155 or shoulder_elbow_wrist_angle2 > 155) and (left_ankle[0] < left_knee[0] < left_hip[0] < left_shoulder[0] or right_ankle[0] > right_knee[0] > right_hip[0] > right_shoulder[0]) and (0 < left_ankle[0] < 1) and (0 < right_ankle[1] < 1) : #Changer conditions d'angles
                        if not timer_started:
                            timer_started = True
                            start_time = time.time()
                    else:
                        timer_started = False
                
                    # Affichage du chrono
                    if timer_started:
                        counter = round(time.time() - start_time, 1)  # Temps écoulé = temps actuel - temps de départ
                        counterBox.configure(text=f'{counter:.2f} s')  # Affichage du temps écoulé avec 2 chiffres après la virgule
                        if counter > PR_rep:
                            PR_rep = counter
                        PRBox.configure(text=f'{PR_rep} s')  # Affichage du temps écoulé avec 2 chiffres après la virgule

    ########################################################################################################
    ########################################################################################################
                                        # Section Jauge d'amplitude et traitement de l'image
    ########################################################################################################
    ######################################################################################################## 

                    # Draw amplitude gauge
                    gauge_value = abs((knee_foot_hip_angle)+(knee_hip_shoulder_angle)) / (320)
                    gauge_value = np.clip(gauge_value, 0, 1)
                    gauge_height = int(200 * gauge_value)

                    cv2.rectangle(image, (20, 300), (60, 300 - gauge_height), (0, 255, 0), -1)
                    cv2.rectangle(image, (20, 100), (60, 300), (255, 255, 255), 2)
                    cv2.putText(image, "% Alignement", (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                except:
                    pass

                # Display reps and stage
                cv2.rectangle(image, (0, 0), (225, 73), (0, 125, 255), -1)
                cv2.putText(image, 'Hold', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                counterBox.configure(text=str(counter))

    # Update alignment bars
                knee_foot_hip_color = "#00FF00" if 160 <= knee_foot_hip_angle <= 180 else "#FF0000"
                knee_hip_shoulder_color = "#00FF00" if 160 <= knee_hip_shoulder_angle <= 180 else "#FF0000"

                # Draw connections with color based on alignment
                connections_color = (0, 255, 0) if knee_foot_hip_color == "#00FF00" and knee_hip_shoulder_color == "#00FF00" else (255, 0, 0)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=connections_color, thickness=2, circle_radius=2),
                                            mp_drawing.DrawingSpec(color=connections_color, thickness=2, circle_radius=2))

                out.write(image)


            # Redimensionner l'image pour qu'elle corresponde au cadre tout en gardant les proportions
            frame_width, frame_height = 480, 480 # Taille définie pour l'affichage
            height, width, _ = image.shape
            scale = min(frame_width / width, frame_height / height)
            resized_width, resized_height = int(width * scale), int(height * scale)

            image_resized = cv2.resize(image, (resized_width, resized_height))
            img = Image.fromarray(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            lmain.imgtk = imgtk
            lmain.configure(image=imgtk)
            lmain.update()

            # Update the progress bar
            progress = (frame_num / total_frames) * 100
            progress_bar['value'] = progress
            progress_bar.update_idletasks()  # Force the UI to update

    

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def main():
        global lmain,counterBox,PRBox, progress_bar
        """Main function to run the pose detection and UI update loop."""
        window, lmain, counterBox, PRBox, progress_bar = setup_ui()
        window.mainloop()

    if __name__ == "__main__":
        main()

def code_upload_fl():
    # Initialize MediaPipe pose and drawing utilities
    # Initialize MediaPipe pose and drawing utilities
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Initialize global variables
    global counter, stage, PR_rep, series_data, current_series, timer_started, start_time
    counter = 0
    stage = None
    PR_rep = 0
    series_data = []  # List to store series and their repetitions
    current_series = 1
    timer_started = False
    start_time = 0
    video_path = ""
    angle_data = []

    def save_to_excel():
        """Save the series data to an Excel file."""  
        df = pd.DataFrame(angle_data, columns=['Time', 'left_knee_foot_hip_angle','left_knee_hip_shoulder_angle','right_knee_hip_shoulder_angle2','right_knee_foot_hip_angle2'])
        file_path = 'output.xlsx'
        try:
            df.to_excel(file_path, index=False)
            print(f"DataFrame saved to {file_path}")
            df.to_csv('repetitions.csv', index=False)
            os.system(f'start excel "{file_path}"')
        except PermissionError:
            print(f"Permission denied: Unable to save {file_path}. Please close the file if it is open and try again.")

    def setup_ui():
        """Setup the main application window and UI elements."""
        global table
        window = tk.Toplevel()
        window.geometry("1280x720")
        window.title("Front-lever timer upload video")
        ck.set_appearance_mode("dark")

        # UI Elements
        classLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10, text='Statistics : ')
        classLabel.place(x=10, y=41)

        counterLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10, text='Front Lever hold')
        counterLabel.place(x=160, y=1)

        PRLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10, text='PR front_lever')
        PRLabel.place(x=160, y=580)

        counterBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="#ff7b00", text='0')
        counterBox.place(x=160, y=41)

        PRBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="Green", text='0')
        PRBox.place(x=160, y=620)

        resetButton = ck.CTkButton(window, text='RESET', command=reset_counter, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="red")
        resetButton.place(x=10, y=620)

        prResetButton = ck.CTkButton(window, text='RESET PR', command=lambda: reset_pr_time(PRBox), height=40, width=120, font=("Arial", 20), text_color="white", fg_color="red")
        prResetButton.place(x=300, y=620)

        saveButton = ck.CTkButton(window, text='SAVE TO EXCEL', command=save_to_excel, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="green")
        saveButton.place(x=575, y=500)

        btn_signaler = ck.CTkButton(window, text="Convert .mov in .mp4 for analysis", command=open_link2, height=40, width=200, text_color="white", fg_color="red")
        btn_signaler.place(x=575, y=20)

        # Button to select video
        videoButton = ck.CTkButton(window, text='Select Video', command=select_video, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
        videoButton.place(x=600, y=620)

        # Button to process video
        processButton = ck.CTkButton(window, text='Process Video', command=lambda: process_video(counterBox, PRBox, lmain), height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
        processButton.place(x=750, y=620)

        frame = tk.Frame(window, height=480, width=480)
        frame.place(x=10, y=90)
        lmain = tk.Label(frame)
        lmain.place(x=0, y=0)


        # Progress Bar
        progress_bar = ttk.Progressbar(window, orient='horizontal', length=400, mode='determinate')
        progress_bar.place(x=480, y=300)

        return window, lmain, counterBox, PRBox, progress_bar

    def select_video():
        """Select a video file."""
        global video_path
        video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])

    def reset_counter(): 
        """Reset the counter and timer to zero."""
        global counter
        counter = 0 

    def reset_pr_time(PRBox):
        """Reset the PR rep and update the display."""
        global PR_rep
        PR_rep = 0
        PRBox.configure(text='0')

    def calculate_angle(a, b, c):
        """Calculate the angle between three points."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return round(angle, 2)

    def process_video(counterBox,PRBox, lmain):
        """Process the selected video and save the repetitions."""
        global video_path, counter, PR_rep, series_data, current_series, progress_bar
        if not video_path:
            return

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            for frame_num in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                try:
                    landmarks = results.pose_landmarks.landmark
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    left_hand = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    right_hand = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    left_mouth = [landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].x, landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].y]
                    right_mouth = [landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].x, landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].y]
                    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]


                    knee_foot_hip_angle = calculate_angle(left_ankle, left_knee, left_hip)
                    knee_hip_shoulder_angle = calculate_angle(left_knee, left_hip, left_shoulder)
                    knee_hip_shoulder_angle2 = calculate_angle(right_knee, right_hip, right_shoulder)
                    knee_foot_hip_angle2 = calculate_angle(right_ankle, right_knee, right_hip)
                    left_shoulder_elbow_wrist_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                    right_shoulder_elbow_wrist_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

                    # Enregistrer l'angle et le timestamp
                    angle_data.append((time.time(), knee_foot_hip_angle,knee_hip_shoulder_angle,knee_hip_shoulder_angle2,knee_foot_hip_angle2))
                    print(angle_data)

    ########################################################################################################
    ########################################################################################################
                                        # Section conditions pour compteur
    ########################################################################################################
    ########################################################################################################                                
                    # Condition pour démarrer le chrono pour le front-lever
                    global timer_started, start_time, PR_rep
                    if (left_wrist[1] < left_elbow[1] < left_shoulder[1] or right_wrist[1] < right_elbow[1] < right_shoulder[1]) and (knee_hip_shoulder_angle > 155 and knee_foot_hip_angle > 155) or (knee_hip_shoulder_angle2 > 155 and knee_foot_hip_angle2 > 155) > 155 and (left_ankle[0] < left_knee[0] < left_hip[0] < left_shoulder[0] or right_ankle[0] > right_knee[0] > right_hip[0] > right_shoulder[0]) and (0 < left_ankle[0] < 1) and (0 < right_ankle[1] < 1) and (left_shoulder_elbow_wrist_angle >155 or right_shoulder_elbow_wrist_angle > 155) : #Changer conditions d'angles
                        if not timer_started:
                            timer_started = True
                            start_time = time.time()
                    else:
                        timer_started = False
                
                    # Affichage du chrono
                    if timer_started:
                        counter = round(time.time() - start_time, 1)  # Temps écoulé = temps actuel - temps de départ
                        counterBox.configure(text=f'{counter:.2f} s')  # Affichage du temps écoulé avec 2 chiffres après la virgule
                        if counter > PR_rep:
                            PR_rep = counter
                        PRBox.configure(text=f'{PR_rep} s') 

    ########################################################################################################
    ########################################################################################################
                                        # Section Jauge d'amplitude et traitement de l'image
    ########################################################################################################
    ######################################################################################################## 

                    # Draw amplitude gauge
                    gauge_value = abs((knee_foot_hip_angle)+(knee_hip_shoulder_angle)) / (320)
                    gauge_value = np.clip(gauge_value, 0, 1)
                    gauge_height = int(200 * gauge_value)

                    cv2.rectangle(image, (20, 300), (60, 300 - gauge_height), (0, 255, 0), -1)
                    cv2.rectangle(image, (20, 100), (60, 300), (255, 255, 255), 2)
                    cv2.putText(image, "% Alignement", (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                except:
                    pass

                # Display reps and stage
                cv2.rectangle(image, (0, 0), (225, 73), (0, 125, 255), -1)
                cv2.putText(image, 'Hold', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                counterBox.configure(text=str(counter))

    # Update alignment bars
                knee_foot_hip_color = "#00FF00" if 160 <= knee_foot_hip_angle <= 180 else "#FF0000"
                knee_hip_shoulder_color = "#00FF00" if 160 <= knee_hip_shoulder_angle <= 180 else "#FF0000"

                # Draw connections with color based on alignment
                connections_color = (0, 255, 0) if knee_foot_hip_color == "#00FF00" and knee_hip_shoulder_color == "#00FF00" else (255, 0, 0)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=connections_color, thickness=2, circle_radius=2),
                                            mp_drawing.DrawingSpec(color=connections_color, thickness=2, circle_radius=2))

                out.write(image)


            # Redimensionnement de l'image pour qu'elle corresponde au cadre de l'interface tout en gardant les proportions
            frame_width, frame_height = 480, 480 # Taille définie pour l'affichage
            height, width, _ = image.shape
            scale = min(frame_width / width, frame_height / height)
            resized_width, resized_height = int(width * scale), int(height * scale)

            image_resized = cv2.resize(image, (resized_width, resized_height))
            img = Image.fromarray(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            lmain.imgtk = imgtk
            lmain.configure(image=imgtk)
            lmain.update()

            # Update the progress bar
            progress = (frame_num / total_frames) * 100
            progress_bar['value'] = progress
            progress_bar.update_idletasks()  # Force the UI to update

    

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def main():
        global lmain,counterBox,PRBox, progress_bar
        """Main function to run the pose detection and UI update loop."""
        window, lmain, counterBox, PRBox, progress_bar = setup_ui()
        window.mainloop()

    if __name__ == "__main__":
        main()

def code_upload_tractions():
    # Initialize MediaPipe pose and drawing utilities
    global counter, stage, PR_rep, series_data, current_series, table, video_path
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Initialize global variables
    counter = 0 
    stage = "up"
    PR_rep = 0
    series_data = []  # List to store series and their repetitions
    current_series = 1
    video_path = ""
    angle_data = []

    def save_to_excel():
        """Save the series data to an Excel file."""  
        df = pd.DataFrame(angle_data, columns=['Time', 'Angle'])
        file_path = 'output.xlsx'
        try:
            df.to_excel(file_path, index=False)
            print(f"DataFrame saved to {file_path}")
            df.to_csv('repetitions.csv', index=False)
            os.system(f'start excel "{file_path}"')
        except PermissionError:
            print(f"Permission denied: Unable to save {file_path}. Please close the file if it is open and try again.")

    def setup_ui():
        """Setup the main application window and UI elements."""
        global table
        window = tk.Toplevel()
        window.geometry("1280x720")
        window.title("Pull ups counter with table series")
        ck.set_appearance_mode("dark")

        # UI Elements
        classLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10, text='Statistics : ')
        classLabel.place(x=10, y=41)

        counterLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10, text='REPS')
        counterLabel.place(x=160, y=1)

        PRLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10, text='PR pull ups')
        PRLabel.place(x=160, y=580)

        counterBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="#ff7b00", text='0')
        counterBox.place(x=160, y=41)

        PRBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="Green", text='0')
        PRBox.place(x=160, y=620)

        resetButton = ck.CTkButton(window, text='RESET', command=reset_counter, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="red")
        resetButton.place(x=10, y=620)

        prResetButton = ck.CTkButton(window, text='RESET PR', command=lambda: reset_pr_time(PRBox), height=40, width=120, font=("Arial", 20), text_color="white", fg_color="red")
        prResetButton.place(x=300, y=620)

        saveButton = ck.CTkButton(window, text='SAVE TO EXCEL', command=save_to_excel, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="green")
        saveButton.place(x=575, y=500)

        # Button to store reps in the series table
        setButton = ck.CTkButton(window, text='SET', command=lambda: store_series(counterBox), height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
        setButton.place(x=450, y=620)

        # Button to select video
        videoButton = ck.CTkButton(window, text='Select Video', command=select_video, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
        videoButton.place(x=600, y=620)

        # Button to process video
        processButton = ck.CTkButton(window, text='Process Video', command=lambda: process_video(counterBox, PRBox, lmain), height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
        processButton.place(x=750, y=620)

        btn_signaler = ck.CTkButton(window, text="Convert .mov in .mp4 for analysis", command=open_link2, height=40, width=200, text_color="white", fg_color="red")
        btn_signaler.place(x=575, y=20)

        # Table for series on the right
        tableFrame = tk.Frame(window, bg="black", width=300, height=600)
        tableFrame.place(x=900, y=10)
        tableLabel = ck.CTkLabel(tableFrame, text="Series et reps", font=("Arial", 16), text_color="white")
        tableLabel.pack()

        table = tk.Text(tableFrame, width=30, height=35, bg="white", fg="black", font=("Arial", 12))
        table.pack()

        frame = tk.Frame(window, height=480, width=480)
        frame.place(x=10, y=90)
        lmain = tk.Label(frame)
        lmain.place(x=0, y=0)


        # Progress Bar
        progress_bar = ttk.Progressbar(window, orient='horizontal', length=400, mode='determinate')
        progress_bar.place(x=480, y=300)

        # Bind the double-click event to the table
        table.bind("<Double-1>", on_double_click)

        return window, lmain, counterBox, PRBox, table, progress_bar

    def select_video():
        """Select a video file."""
        global video_path
        video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])

    def reset_counter(): 
        """Reset the counter and timer to zero."""
        global counter
        counter = 0 

    def reset_pr_time(PRBox):
        """Reset the PR rep and update the display."""
        global PR_rep
        PR_rep = 0
        PRBox.configure(text='0')

    def calculate_angle(a, b, c):
        """Calculate the angle between three points."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return round(angle, 2)

    def update_line_color(image, left_hand, right_hand, left_mouth, right_mouth, shoulder_left, shoulder_right):
        """Update the line color based on positions."""
        global color, en_haut
        color = (255, 0, 0)  # Default to red
        en_haut = False

        if left_hand[1] > left_mouth[1] and right_hand[1] > right_mouth[1]:
            color = (0, 255, 0)  # Set to green if condition met
            en_haut = True

        if left_hand[1] < shoulder_left[1] and right_hand < shoulder_right:
            cv2.line(image, 
                    (int(left_hand[0]), int(left_hand[1])), 
                    (int(right_hand[0]), int(right_hand[1])), 
                    color, 4)

    def update_table():
        """Update the table with the series data and store it in an Excel file."""
        global series_data, table
        table.delete('1.0', tk.END)  # Clear the table
        
        # Update the table in the GUI
        for series, reps in series_data:
            table.insert(tk.END, f"Série {series}: {reps} reps\n")

    def on_double_click(event):
        """Handle double-click event to modify reps."""
        global series_data, table
        # Get the line number where the double-click occurred
        line_index = table.index("@%s,%s" % (event.x, event.y)).split('.')[0]
        line_index = int(line_index) - 1  # Convert to zero-based index
        
        if 0 <= line_index < len(series_data):
            # Prompt the user to enter a new value for reps
            new_reps = simpledialog.askinteger("Input", f"Enter new repetitions for Série {series_data[line_index][0]}:", minvalue=0)
            if new_reps is not None:
                # Update the series_data with the new value
                series_data[line_index] = (time.time(),series_data[line_index][0], new_reps)
                # Update the table
                update_table()

    def store_series(counterBox):
        """Store the current counter value as a series and reset the counter."""
        global series_data, counter, current_series
        reps = int(counterBox.cget("text"))
        series_data.append((current_series, reps))
        update_table()
        current_series += 1
        reset_counter()
        counterBox.configure(text='0')


    def process_video(counterBox, PRBox, lmain):
        """Process the selected video and save the repetitions."""
        global video_path, counter, PR_rep, series_data, current_series, progress_bar
        if not video_path:
            return

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            for frame_num in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                try:
                    landmarks = results.pose_landmarks.landmark
                    shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    left_hand = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    right_hand = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    left_mouth = [landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].x, landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].y]
                    right_mouth = [landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].x, landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].y]
                    
                    angle = calculate_angle(shoulder_left, elbow, wrist)
                    update_line_color(image, left_hand, right_hand, left_mouth, right_mouth, shoulder_left, shoulder_right)

                    # Enregistrer l'angle et le timestamp
                    angle_data.append((time.time(), angle))
                    print(angle_data)

                    global color, en_haut

                    color = (255, 0, 0)
                    en_haut = False

                    # Dessiner la ligne entre les mains
                    if left_hand[1] < elbow[1] and right_hand[1] < elbow[1]:
                        # Vérifier si la bouche est au-dessus de la ligne
                        if left_mouth[1] < min(left_hand[1], right_hand[1]) or right_mouth[1] < min(left_hand[1], right_hand[1]): # Peut-être changer la condition pour accepter une marge d'erreur
                            en_haut = True
                            color = (0,255,0)

                    global stage, counter, PR_rep
                    if hip[1] > shoulder_left[1] and left_hand[1] < shoulder_left[1]:
                        if angle > 160:  # elbow
                            stage = "down"
                        if en_haut and stage == 'down' and left_hand[1] < elbow[1]:
                            stage = "up"
                            counter += 1
                            print(counter)
                    if counter > PR_rep:
                        PR_rep = counter
                        PRBox.configure(text=str(PR_rep))

                    # Draw amplitude gauge
                    min_angle, max_angle = 90, 160
                    gauge_value = abs(max_angle - angle) / (max_angle - min_angle)
                    gauge_value = np.clip(gauge_value, 0, 1)
                    gauge_height = int(200 * gauge_value)

                    cv2.rectangle(image, (20, 300), (60, 300 - gauge_height), (0, 255, 0), -1)
                    cv2.rectangle(image, (20, 100), (60, 300), (255, 255, 255), 2)
                    cv2.putText(image, 'Range of motion', (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1) 
                except:
                    pass

                # Display reps and stage
                cv2.rectangle(image, (0, 0), (225, 73), (0, 125, 255), -1)
                cv2.putText(image, 'REPS', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, 'STAGE', (65, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, stage, (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), 
                                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))               

                out.write(image)


            # Redimensionner l'image pour qu'elle corresponde au cadre tout en gardant les proportions
            frame_width, frame_height = 480, 480 # Taille définie pour l'affichage
            height, width, _ = image.shape
            scale = min(frame_width / width, frame_height / height)
            resized_width, resized_height = int(width * scale), int(height * scale)

            image_resized = cv2.resize(image, (resized_width, resized_height))
            img = Image.fromarray(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            lmain.imgtk = imgtk
            lmain.configure(image=imgtk)
            lmain.update()

            # Update the progress bar
            progress = (frame_num / total_frames) * 100
            progress_bar['value'] = progress
            progress_bar.update_idletasks()  # Force the UI to update

    

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def main():
        global table,lmain,counterBox,PRBox, progress_bar
        """Main function to run the pose detection and UI update loop."""
        window, lmain, counterBox, PRBox, table, progress_bar = setup_ui()
        window.mainloop()

    if __name__ == "__main__":
        main()

def code_upload_squat():
    # Initialize MediaPipe pose and drawing utilities
    global counter, stage, PR_rep, video_path
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Initialize global variables
    counter = 0 
    stage = "up"
    PR_rep = 0
    video_path = ""
    angle_data = []

    def save_to_excel():
        """Save the series data to an Excel file."""  
        df = pd.DataFrame(angle_data, columns=['Time', 'Angle jambe gauche', 'Angle jambe droite'])
        file_path = 'output.xlsx'
        try:
            df.to_excel(file_path, index=False)
            print(f"DataFrame saved to {file_path}")
            df.to_csv('repetitions.csv', index=False)
            os.system(f'start excel "{file_path}"')
        except PermissionError:
            print(f"Permission denied: Unable to save {file_path}. Please close the file if it is open and try again.")

    def setup_ui():
        """Setup the main application window and UI elements."""
        global table
        window = tk.Toplevel()
        window.geometry("1280x720")
        window.title("Squat Counter upload video")
        ck.set_appearance_mode("dark")

        # UI Elements
        classLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10, text='Statistics : ')
        classLabel.place(x=10, y=41)

        counterLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10, text='REPS')
        counterLabel.place(x=160, y=1)

        PRLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10, text='PR squats')
        PRLabel.place(x=160, y=580)

        counterBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="#ff7b00", text='0')
        counterBox.place(x=160, y=41)

        PRBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="Green", text='0')
        PRBox.place(x=160, y=620)

        resetButton = ck.CTkButton(window, text='RESET', command=reset_counter, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="red")
        resetButton.place(x=10, y=620)

        prResetButton = ck.CTkButton(window, text='RESET PR', command=lambda: reset_pr_time(PRBox), height=40, width=120, font=("Arial", 20), text_color="white", fg_color="red")
        prResetButton.place(x=300, y=620)

        saveButton = ck.CTkButton(window, text='SAVE TO EXCEL', command=save_to_excel, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="green")
        saveButton.place(x=575, y=500)

        btn_signaler = ck.CTkButton(window, text="Convert .mov in .mp4 for analysis", command=open_link2, height=40, width=200, text_color="white", fg_color="red")
        btn_signaler.place(x=575, y=20)

        # Button to select video
        videoButton = ck.CTkButton(window, text='Select Video', command=select_video, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
        videoButton.place(x=600, y=620)

        # Button to process video
        processButton = ck.CTkButton(window, text='Process Video', command=lambda: process_video(counterBox, PRBox, lmain), height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
        processButton.place(x=750, y=620)

        frame = tk.Frame(window, height=480, width=480)
        frame.place(x=10, y=90)
        lmain = tk.Label(frame)
        lmain.place(x=0, y=0)


        # Progress Bar
        progress_bar = ttk.Progressbar(window, orient='horizontal', length=400, mode='determinate')
        progress_bar.place(x=480, y=300)

        return window, lmain, counterBox, PRBox, progress_bar

    def select_video():
        """Select a video file."""
        global video_path
        video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])

    def reset_counter(): 
        """Reset the counter and timer to zero."""
        global counter
        counter = 0 

    def reset_pr_time(PRBox):
        """Reset the PR rep and update the display."""
        global PR_rep
        PR_rep = 0
        PRBox.configure(text='0')

    def calculate_angle(a, b, c):
        """Calculate the angle between three points."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return round(angle, 2)

    def process_video(counterBox,PRBox, lmain):
        """Process the selected video and save the repetitions."""
        global video_path, counter, PR_rep, series_data, current_series, progress_bar
        if not video_path:
            return

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            for frame_num in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                try:
                    landmarks = results.pose_landmarks.landmark
                    shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    left_hand = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    right_hand = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    left_mouth = [landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].x, landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].y]
                    right_mouth = [landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].x, landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].y]
                    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]


                    angle = calculate_angle(left_ankle, left_knee, left_hip)
                    angle2 = calculate_angle(right_ankle, right_knee, right_hip)

                    # Enregistrer l'angle et le timestamp
                    angle_data.append((time.time(), angle,angle2))
                    print(angle_data)

    ########################################################################################################
    ########################################################################################################
                                        # Section conditions pour compteur
    ########################################################################################################
    ########################################################################################################                                
                    global stage, counter, PR_rep
                    if angle > 155 or angle2 > 155 and (0 < left_ankle[0] < 1) and (0 < right_ankle[1] < 1) and (0 < left_ankle[0] < 1) and (0 < right_ankle[1] < 1):
                        stage = "up"
                    if (angle < 85 or angle2 < 85) and stage == 'up':
                        stage = "down"
                        counter += 1
                        if counter > PR_rep:
                            PR_rep = counter
                            PRBox.configure(text=str(PR_rep))
                    counterBox.configure(text=str(counter))

    ########################################################################################################
    ########################################################################################################
                                        # Section Jauge d'amplitude et traitement de l'image
    ########################################################################################################
    ######################################################################################################## 

                    # Draw amplitude gauge
                    min_angle, max_angle = 90, 160
                    gauge_value = (angle - min_angle) / (max_angle - min_angle)
                    gauge_value = np.clip(gauge_value, 0, 1)
                    gauge_height = int(200 * gauge_value)

                    cv2.rectangle(image, (20, 300), (60, 300 - gauge_height), (0, 255, 0), -1)
                    cv2.rectangle(image, (20, 100), (60, 300), (255, 255, 255), 2)
                    cv2.putText(image, 'Amplitude', (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1) 
                except:
                    pass

                # Display reps and stage
                cv2.rectangle(image, (0, 0), (225, 73), (0, 125, 255), -1)
                cv2.putText(image, 'REPS', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, 'STAGE', (65, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, stage, (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                # Tracé des points et des lignes de détection du modèle
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), 
                                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))               

                out.write(image)


            # Redimensionner l'image pour qu'elle corresponde au cadre tout en gardant les proportions
            frame_width, frame_height = 480, 480 # Taille définie pour l'affichage
            height, width, _ = image.shape
            scale = min(frame_width / width, frame_height / height)
            resized_width, resized_height = int(width * scale), int(height * scale)

            image_resized = cv2.resize(image, (resized_width, resized_height))
            img = Image.fromarray(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            lmain.imgtk = imgtk
            lmain.configure(image=imgtk)
            lmain.update()

            # Update the progress bar
            progress = (frame_num / total_frames) * 100
            progress_bar['value'] = progress
            progress_bar.update_idletasks()  # Force the UI to update

    

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def main():
        global lmain,counterBox,PRBox, progress_bar
        """Main function to run the pose detection and UI update loop."""
        window, lmain, counterBox, PRBox, progress_bar = setup_ui()
        window.mainloop()

    if __name__ == "__main__":
        main()

def code_traction():
    # Initialize MediaPipe pose and drawing utilities
    global counter,stage,PR_rep,series_data,current_series,table
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Initialize global variables
    counter = 0 
    stage = None
    PR_rep = 0
    series_data = []  # List to store series and their repetitions
    current_series = 1

    def setup_ui():
        """Setup the main application window and UI elements."""
        global table
        window = tk.Toplevel()
        window.geometry("1280x720")
        window.title("Pull ups counter with series table")
        ck.set_appearance_mode("dark")

        # UI Elements
        classLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10, text='Statistics : ')
        classLabel.place(x=10, y=41)

        counterLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10, text='REPS')
        counterLabel.place(x=160, y=1)

        PRLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10, text='PR pull ups')
        PRLabel.place(x=160, y=580)

        counterBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="#ff7b00", text='0')
        counterBox.place(x=160, y=41)

        PRBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="Green", text='0')
        PRBox.place(x=160, y=620)

        resetButton = ck.CTkButton(window, text='RESET', command=reset_counter, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="red")
        resetButton.place(x=10, y=620)

        prResetButton = ck.CTkButton(window, text='RESET PR', command=lambda: reset_pr_time(PRBox), height=40, width=120, font=("Arial", 20), text_color="white", fg_color="red")
        prResetButton.place(x=300, y=620)

        # Button to store reps in the series table
        setButton = ck.CTkButton(window, text='SET', command=lambda: store_series(counterBox), height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
        setButton.place(x=450, y=620)

        # Table for series on the right
        tableFrame = tk.Frame(window,bg="black", width=300, height=600)
        tableFrame.place(x=900, y=10)
        tableLabel = ck.CTkLabel(tableFrame, text="Series and reps", font=("Arial", 16), text_color="white")
        tableLabel.pack()

        table = tk.Text(tableFrame, width=30, height=35, bg="white", fg="black", font=("Arial", 12))
        table.pack()

        frame = tk.Frame(window,height=480, width=480)
        frame.place(x=10, y=90)
        lmain = tk.Label(frame)
        lmain.place(x=0, y=0)
        # Bind the double-click event to the table
        table.bind("<Double-1>", on_double_click)

        return window, lmain, counterBox, PRBox, table

    def reset_counter(): 
        """Reset the counter and timer to zero."""
        global counter
        counter = 0 

    def reset_pr_time(PRBox):
        """Reset the PR rep and update the display."""
        global PR_rep
        PR_rep = 0
        PRBox.configure(text='0')

    def calculate_angle(a, b, c):
        """Calculate the angle between three points."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return round(angle, 2)

    def update_line_color(image, left_hand, right_hand, left_mouth, right_mouth, shoulder_left,shoulder_right):
        """Update the line color based on positions."""
        global color, en_haut
        color = (255, 0, 0)  # Default to red
        en_haut = False

        if left_hand[1] > left_mouth[1] and right_hand[1] > right_mouth[1]:
            color = (0, 255, 0)  # Set to green if condition met
            en_haut = True

        if left_hand[1] < shoulder_left[1] and right_hand < shoulder_right :
            cv2.line(image, 
                    (int(left_hand[0] * 480)+120, int(left_hand[1] * 480)), 
                    (int(right_hand[0] * 480), int(right_hand[1] * 480)), 
                    color, 4)
        
    def update_table():
        """Update the table with the series data and store it in an Excel file."""
        global series_data, table
        table.delete('1.0', tk.END)  # Clear the table
        
        # Create a DataFrame from the series_data
        df = pd.DataFrame(series_data, columns=['Series', 'Reps'])
        
        # Update the table in the GUI
        for series, reps in series_data:
            table.insert(tk.END, f"Série {series}: {reps} reps\n")

    def on_double_click(event):
        """Handle double-click event to modify reps."""
        global series_data, table
        # Get the line number where the double-click occurred
        line_index = table.index("@%s,%s" % (event.x, event.y)).split('.')[0]
        line_index = int(line_index) - 1  # Convert to zero-based index
        
        if 0 <= line_index < len(series_data):
            # Prompt the user to enter a new value for reps
            new_reps = simpledialog.askinteger("Input", f"Enter new repetitions for Serie {series_data[line_index][0]}:", minvalue=0)
            if new_reps is not None:
                # Update the series_data with the new value
                series_data[line_index] = (series_data[line_index][0], new_reps)
                # Update the table
                update_table()

    def store_series(counterBox):
        """Store the current counter value as a series and reset the counter."""
        global series_data, counter, current_series,stage
        reps = int(counterBox.cget("text"))
        series_data.append((current_series, reps))
        update_table()
        current_series += 1
        reset_counter()
        counterBox.configure(text='0')
        stage = None

    def update_frame(cap, pose, lmain, counterBox, PRBox,table):
        """Update the frame for the video capture and UI."""
        ret, frame = cap.read()
        
        if not ret:
            lmain.after(10, update_frame, cap, pose, lmain, counterBox, PRBox, table)
            return
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        try:
            landmarks = results.pose_landmarks.landmark
            shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            left_hand = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            right_hand = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            left_mouth = [landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].x, landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].y]
            right_mouth = [landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].x, landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].y]
            
            angle = calculate_angle(shoulder_left, elbow, wrist)
            angle_droit = calculate_angle(shoulder_right, right_elbow, right_wrist)
            update_line_color(image, left_hand, right_hand, left_mouth, right_mouth,shoulder_left,shoulder_right)

            global color, en_haut

            color = (255, 0, 0)
            en_haut = False

            # Dessiner la ligne entre les mains
            if left_hand[1] < elbow[1] and right_hand[1] < elbow[1]:
                # Vérifier si la bouche est au-dessus de la ligne
                if left_mouth[1] < min(left_hand[1], right_hand[1]) or right_mouth[1] < min(left_hand[1], right_hand[1]): # Peut-être changer la condition pour accepter une marge d'erreur
                    en_haut = True
                    color = (0,255,0)

            global stage, counter, PR_rep
            if hip[1] > shoulder_left[1] and left_hand[1] < shoulder_left[1] :
                if (angle > 160) or (angle_droit > 160) and stage == 'up' : ## elbow
                    if stage == 'up':
                        counter += 1
                        stage = "down"
                        print(counter)
                    if stage == None :
                        stage = "down"                  
                if en_haut and stage == 'down' and left_hand[1] < elbow[1]:
                    stage = "up"
            if counter > PR_rep:
                PR_rep = counter
                PRBox.configure(text=str(PR_rep))

            # Draw amplitude gauge
            min_angle, max_angle = 90, 160
            gauge_value = abs(max_angle - angle) / (max_angle - min_angle)
            gauge_value = np.clip(gauge_value, 0, 1)
            gauge_height = int(200 * gauge_value)

            cv2.rectangle(image, (20, 300), (60, 300 - gauge_height), (0, 255, 0), -1)
            cv2.rectangle(image, (20, 100), (60, 300), (255, 255, 255), 2)
            cv2.putText(image, 'Range of motion', (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1) 
        except:
            pass

        # Display reps and stage
        cv2.rectangle(image, (0, 0), (225, 73), (0, 125, 255), -1)
        cv2.putText(image, 'REPS', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, 'STAGE', (65, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))               

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        
        counterBox.configure(text=str(counter))
        
        lmain.after(10, update_frame, cap, pose, lmain, counterBox, PRBox, table)

    def main():
        """Main function to run the pose detection and UI update loop."""
        window, lmain, counterBox, PRBox, table = setup_ui()

        cap = cv2.VideoCapture(camera_index)
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            update_frame(cap, pose, lmain, counterBox, PRBox, table)
            window.mainloop()

        cap.release()
        cv2.destroyAllWindows()

    if __name__ == "__main__":
        main()

    pass

def code_pompes():
    # Initialize MediaPipe pose and drawing utilities
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    global counter,stage,PR_rep,series_data,current_series,table
    # Initialize global variables
    counter = 0
    stage = "up" # Si None condition up dans comptage => marche pas
    PR_rep = 0
    series_data = []  # List to store series and their repetitions
    current_series = 1

    class ColorProgressBar(ck.CTkProgressBar): # classe pour la jauge d'amplitude
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set(1)

        def set_value(self, value, color):
            self.set(value)
            self.configure(progress_color=color)

    def setup_ui():
        """Setup the main application window and UI elements."""
        global table
        window = tk.Toplevel()
        window.geometry("1280x720")
        window.title("Push ups counter with series table")
        ck.set_appearance_mode("dark")

        # UI Elements
        classLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10, text='Statistics : ')
        classLabel.place(x=10, y=41)

        counterLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10, text='REPS')
        counterLabel.place(x=160, y=1)

        PRLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10, text='PR push ups')
        PRLabel.place(x=160, y=580)

        counterBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="#ff7b00", text='0')
        counterBox.place(x=160, y=41)

        PRBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="Green", text='0')
        PRBox.place(x=160, y=620)

        resetButton = ck.CTkButton(window, text='RESET', command=reset_counter, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="red")
        resetButton.place(x=10, y=620)

        prResetButton = ck.CTkButton(window, text='RESET PR', command=lambda: reset_pr_time(PRBox), height=40, width=120, font=("Arial", 20), text_color="white", fg_color="red")
        prResetButton.place(x=300, y=620)

        # Button to store reps in the series table
        setButton = ck.CTkButton(window, text='SET', command=lambda: store_series(counterBox), height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
        setButton.place(x=450, y=620)

        # Table for series on the right
        tableFrame = tk.Frame(window,bg="black", width=300, height=600)
        tableFrame.place(x=900, y=10)
        tableLabel = ck.CTkLabel(tableFrame, text="Series et reps", font=("Arial", 16), text_color="white")
        tableLabel.pack()

        table = tk.Text(tableFrame, width=30, height=35, bg="white", fg="black", font=("Arial", 12))
        table.pack()

        frame = tk.Frame(window,height=480, width=480)
        frame.place(x=10, y=90)
        lmain = tk.Label(frame)
        lmain.place(x=0, y=0)

        # Progress bars for alignment
        knee_foot_hip_bar = ColorProgressBar(window, height=20, width=120)
        knee_foot_hip_bar.place(x=500, y=100)
        knee_foot_hip_label = ck.CTkLabel(window, height=20, width=120, font=("Arial", 12), text_color="black", text="feet-knee-hip")
        knee_foot_hip_label.place(x=500, y=80)

        knee_hip_shoulder_bar = ColorProgressBar(window, height=20, width=120)
        knee_hip_shoulder_bar.place(x=500, y=160)
        knee_hip_shoulder_label = ck.CTkLabel(window, height=20, width=120, font=("Arial", 12), text_color="black", text="knee-hip-shoulder")
        knee_hip_shoulder_label.place(x=500, y=140)
        # Bind the double-click event to the table
        table.bind("<Double-1>", on_double_click)

        return window, lmain, counterBox, PRBox, knee_foot_hip_bar, knee_hip_shoulder_bar,table

    def reset_counter():
        """Reset the counter and stage to zero."""
        global counter, stage
        counter = 0
        stage = "down"

    def reset_pr_time(PRBox):
        """Reset the PR rep and update the display."""
        global PR_rep
        PR_rep = 0
        PRBox.configure(text='0')

    def calculate_angle(a, b, c):
        """Calculate the angle between three points."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return round(angle, 2)

    def update_table():
        """Update the table with the series data and store it in an Excel file."""
        global series_data, table
        table.delete('1.0', tk.END)  # Clear the table
        
        # Create a DataFrame from the series_data
        df = pd.DataFrame(series_data, columns=['Series', 'Repetitions'])
        
        # Update the table in the GUI
        for series, reps in series_data:
            table.insert(tk.END, f"Serie {series}: {reps} reps\n")

    def on_double_click(event):
        """Handle double-click event to modify reps."""
        global series_data, table
        # Get the line number where the double-click occurred
        line_index = table.index("@%s,%s" % (event.x, event.y)).split('.')[0]
        line_index = int(line_index) - 1  # Convert to zero-based index
        
        if 0 <= line_index < len(series_data):
            # Prompt the user to enter a new value for reps
            new_reps = simpledialog.askinteger("Input", f"Enter new repetitions for Serie {series_data[line_index][0]}:", minvalue=0)
            if new_reps is not None:
                # Update the series_data with the new value
                series_data[line_index] = (series_data[line_index][0], new_reps)
                # Update the table
                update_table()

    def store_series(counterBox):
        """Store the current counter value as a series and reset the counter."""
        global series_data, counter, current_series
        reps = int(counterBox.cget("text"))
        series_data.append((current_series, reps))
        update_table()
        current_series += 1
        reset_counter()
        counterBox.configure(text='0')

    def update_frame(cap, pose, lmain, counterBox, PRBox, knee_foot_hip_bar, knee_hip_shoulder_bar,table):
        """Update the frame for the video capture and UI."""
        global counter, stage, PR_rep

        ret, frame = cap.read()
        if not ret:
            lmain.after(10, update_frame, cap, pose, lmain, counterBox, PRBox, knee_foot_hip_bar, knee_hip_shoulder_bar,table)
            return

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_foot = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                right_foot = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]


                angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                angle2 = calculate_angle(right_shoulder, right_elbow, right_wrist)
                knee_foot_hip_angle = calculate_angle(left_foot, left_knee, left_hip)
                knee_hip_shoulder_angle = calculate_angle(left_knee, left_hip, left_shoulder)

                # Update stage and counter
                if (left_wrist[1] > left_elbow[1] or right_wrist[1] > right_elbow[1]) and knee_hip_shoulder_angle > 150 and knee_foot_hip_angle > 150 and (left_foot[0] > left_knee[0] > left_hip[0] > left_shoulder[0] or right_foot[0] < right_knee[0] < right_hip[0] < right_shoulder[0]) :
                    if (angle > 150 or angle2 > 150) and (0 < left_foot[0] < 1) and (0 < right_foot[1] < 1):
                        if stage == "up":
                            counter += 1
                            stage = "down"
                    if (angle < 90 or angle2 < 90) and stage == 'down':
                        stage = "up"
                        counterBox.configure(text=str(counter))

                if counter > PR_rep:
                    PR_rep = counter
                    PRBox.configure(text=str(PR_rep))

                # Draw amplitude gauge
                min_angle, max_angle = 90, 160
                gauge_value = (angle - min_angle) / (max_angle - min_angle)
                gauge_value = np.clip(gauge_value, 0, 1)
                gauge_height = int(200 * gauge_value)

                cv2.rectangle(image, (20, 300), (60, 300 - gauge_height), (0, 255, 0), -1)
                cv2.rectangle(image, (20, 100), (60, 300), (255, 255, 255), 2)
                cv2.putText(image, 'Range of motion', (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Update alignment bars
                knee_foot_hip_color = "#00FF00" if 160 <= knee_foot_hip_angle <= 180 else "#FF0000"
                knee_hip_shoulder_color = "#00FF00" if 160 <= knee_hip_shoulder_angle <= 180 else "#FF0000"

                knee_foot_hip_bar.set_value((knee_foot_hip_angle - 160) / 20, knee_foot_hip_color)
                knee_hip_shoulder_bar.set_value((knee_hip_shoulder_angle - 160) / 20, knee_hip_shoulder_color)

                # Draw connections with color based on alignment
                connections_color = (0, 255, 0) if knee_foot_hip_color == "#00FF00" and knee_hip_shoulder_color == "#00FF00" else (255, 0, 0)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=connections_color, thickness=2, circle_radius=2),
                                        mp_drawing.DrawingSpec(color=connections_color, thickness=2, circle_radius=2))

        except Exception as e:
            print(e)

        # Display reps and stage
        cv2.rectangle(image, (0, 0), (225, 73), (0, 125, 255), -1)
        cv2.putText(image, 'REPS', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, 'STAGE', (65, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)

        counterBox.configure(text=str(counter))

        lmain.after(10, update_frame, cap, pose, lmain, counterBox, PRBox, knee_foot_hip_bar, knee_hip_shoulder_bar,table)

    def main():
        """Main function to run the pose detection and UI update loop."""
        window, lmain, counterBox, PRBox, knee_foot_hip_bar, knee_hip_shoulder_bar,table = setup_ui()

        cap = cv2.VideoCapture(camera_index)
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            update_frame(cap, pose, lmain, counterBox, PRBox, knee_foot_hip_bar, knee_hip_shoulder_bar,table)
            window.mainloop()

        cap.release()
        cv2.destroyAllWindows()

    if __name__ == "__main__":
        main()
    pass

def code_squats():
     # Initialize MediaPipe pose and drawing utilities
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Initialize global variables
    global counter, stage, PR_rep, series_data, current_series, table
    counter = 0 
    stage = "up"
    PR_rep = 0
    series_data = []  # List to store series and their repetitions
    current_series = 1

    def setup_ui():
        """Setup the main application window and UI elements."""
        global table
        window = tk.Toplevel()
        window.geometry("1280x720")
        window.title("Squats counter with series table")
        ck.set_appearance_mode("dark")

        # UI Elements
        classLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10, text='Statistics : ')
        classLabel.place(x=10, y=41)

        counterLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10, text='REPS')
        counterLabel.place(x=160, y=1)

        PRLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10, text='PR squats')
        PRLabel.place(x=160, y=580)

        counterBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="#ff7b00", text='0')
        counterBox.place(x=160, y=41)

        PRBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="Green", text='0')
        PRBox.place(x=160, y=620)

        resetButton = ck.CTkButton(window, text='RESET', command=reset_counter, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="red")
        resetButton.place(x=10, y=620)

        prResetButton = ck.CTkButton(window, text='RESET PR', command=lambda: reset_pr_time(PRBox), height=40, width=120, font=("Arial", 20), text_color="white", fg_color="red")
        prResetButton.place(x=300, y=620)

        # Button to store reps in the series table
        setButton = ck.CTkButton(window, text='SET', command=lambda: store_series(counterBox), height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
        setButton.place(x=450, y=620)

        # Table for series on the right
        tableFrame = tk.Frame(window,bg="black", width=300, height=600)
        tableFrame.place(x=900, y=10)
        tableLabel = ck.CTkLabel(tableFrame, text="Series et reps", font=("Arial", 16), text_color="white")
        tableLabel.pack()

        table = tk.Text(tableFrame, width=30, height=35, bg="white", fg="black", font=("Arial", 12))
        table.pack()

        frame = tk.Frame(window,height=480, width=480)
        frame.place(x=10, y=90)
        lmain = tk.Label(frame)
        lmain.place(x=0, y=0)
        # Bind the double-click event to the table
        table.bind("<Double-1>", on_double_click)        

        return window, lmain, counterBox, PRBox, table

    def reset_counter(): 
        global counter
        counter = 0 

    def reset_pr_time(PRBox):
        """Reset the PR rep and update the display."""
        global PR_rep
        PR_rep = 0
        PRBox.configure(text='0')

    def calculate_angle(a, b, c):
        """Calculate the angle between three points."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return round(angle, 2)

    def update_table():
        """Update the table with the series data and store it in an Excel file."""
        global series_data, table
        table.delete('1.0', tk.END)  # Clear the table
        
        # Create a DataFrame from the series_data
        df = pd.DataFrame(series_data, columns=['Series', 'reps'])
        
        # Update the table in the GUI
        for series, reps in series_data:
            table.insert(tk.END, f"Serie {series}: {reps} reps\n")

    def on_double_click(event):
        """Handle double-click event to modify reps."""
        global series_data, table
        # Get the line number where the double-click occurred
        line_index = table.index("@%s,%s" % (event.x, event.y)).split('.')[0]
        line_index = int(line_index) - 1  # Convert to zero-based index
        
        if 0 <= line_index < len(series_data):
            # Prompt the user to enter a new value for reps
            new_reps = simpledialog.askinteger("Input", f"Enter new repetitions for Serie {series_data[line_index][0]}:", minvalue=0)
            if new_reps is not None:
                # Update the series_data with the new value
                series_data[line_index] = (series_data[line_index][0], new_reps)
                # Update the table
                update_table()

    def store_series(counterBox):
        """Store the current counter value as a series and reset the counter."""
        global series_data, counter, current_series
        reps = int(counterBox.cget("text"))
        series_data.append((current_series, reps))
        update_table()
        current_series += 1
        reset_counter()
        counterBox.configure(text='0')

    def update_frame(cap, pose, lmain, counterBox, PRBox, table):
        ret, frame = cap.read()
        
        if not ret:
            lmain.after(10, update_frame, cap, pose, lmain, counterBox, PRBox, table)
            return
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        try:
            landmarks = results.pose_landmarks.landmark
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            left_foot = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            right_foot = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            
            angle = calculate_angle(hip, knee, ankle)
            angle2 = calculate_angle(right_hip, right_knee, right_ankle)
            
            global stage, counter, PR_rep
            if angle > 155 or angle2 > 155 and (0 < left_foot[0] < 1) and (0 < right_foot[1] < 1) and (0 < left_foot[0] < 1) and (0 < right_foot[1] < 1):
                stage = "up"
            if (angle < 85 or angle2 < 85) and stage == 'up':
                stage = "down"
                counter += 1
                if counter > PR_rep:
                    PR_rep = counter
                    PRBox.configure(text=str(PR_rep))
            counterBox.configure(text=str(counter))

            # Draw amplitude gauge
            min_angle, max_angle = 90, 160
            gauge_value = (angle - min_angle) / (max_angle - min_angle)
            gauge_value = np.clip(gauge_value, 0, 1)
            gauge_height = int(200 * gauge_value)

            cv2.rectangle(image, (20, 300), (60, 300 - gauge_height), (0, 255, 0), -1)
            cv2.rectangle(image, (20, 100), (60, 300), (255, 255, 255), 2)
            cv2.putText(image, 'Range of motion', (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
        except:
            pass

        # Display reps and stage
        cv2.rectangle(image, (0, 0), (225, 73), (0, 125, 255), -1)
        cv2.putText(image, 'REPS', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, 'STAGE', (65, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))               

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        
        lmain.after(10, update_frame, cap, pose, lmain, counterBox, PRBox,table)

    def main():
        window, lmain, counterBox, PRBox, table = setup_ui()
        cap = cv2.VideoCapture(camera_index)
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            update_frame(cap, pose, lmain, counterBox, PRBox, table)
            window.mainloop()
        cap.release()
        cv2.destroyAllWindows()

    if __name__ == "__main__":
        main()

    pass

def code_HSPU():
    # Initialize MediaPipe pose and drawing utilities
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    global counter, stage, timer_started, start_time, PR_time, series_data, current_series, table
    # Initialize global variables
    counter = 0 
    stage = None
    timer_started = False
    start_time = 0
    PR_time = 0
    series_data = []  # List to store series and their repetitions
    current_series = 1

    def setup_ui():
        """Setup the main application window and UI elements."""
        global table, window
        window = tk.Toplevel()
        window.geometry("1280x720")
        window.title("HSPU counter and HS timer with series table") 
        ck.set_appearance_mode("dark")

        classLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10)
        classLabel.place(x=10, y=41)
        classLabel.configure(text='Statistics : ') 

        counterLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10)
        counterLabel.place(x=160, y=1)
        counterLabel.configure(text='REPS') 

        probLabel  = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10)
        probLabel.place(x=300, y=1)
        probLabel.configure(text='TIMER') 

        PRLabel  = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10)
        PRLabel.place(x=160, y=580)
        PRLabel.configure(text='PR HS') 

        counterBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="#ff7b00")
        counterBox.place(x=160, y=41)
        counterBox.configure(text='0') 

        probBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="#ff7b00")
        probBox.place(x=300, y=41)
        probBox.configure(text='0') 

        PRBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="Green")
        PRBox.place(x=160, y=620)
        PRBox.configure(text='0')

        button = ck.CTkButton(window, text='RESET', command=reset_counter, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="red")
        button.place(x=10, y=620)

        pr_reset_button = ck.CTkButton(window, text='RESET PR', command=lambda: reset_pr_time(probBox, PRBox), height=40, width=120, font=("Arial", 20), text_color="white", fg_color="red")
        pr_reset_button.place(x=300, y=620)

        # Button to store reps in the series table
        setButton = ck.CTkButton(window, text='SET', command=lambda: store_series(counterBox), height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
        setButton.place(x=450, y=620)

        # Table for series on the right
        tableFrame = tk.Frame(window, bg="black", width=300, height=600)
        tableFrame.place(x=900, y=10)
        tableLabel = ck.CTkLabel(tableFrame, text="Series et reps", font=("Arial", 16), text_color="white")
        tableLabel.pack()

        table = tk.Text(tableFrame, width=30, height=35, bg="white", fg="black", font=("Arial", 12))
        table.pack()

        frame = tk.Frame(window,height=480, width=480)
        frame.place(x=10, y=90)
        lmain = tk.Label(frame)
        lmain.place(x=0, y=0)
        # Bind the double-click event to the table
        table.bind("<Double-1>", on_double_click)

        return window, lmain, counterBox, probBox, PRBox,table

    def reset_counter(): 
        """Reset the counter and timer to zero."""
        global counter, timer_started, start_time, PR_time
        counter = 0 
        timer_started = False
        start_time = 0

    def reset_pr_time(probBox, PRBox):
        """Reset the PR time and update the display."""
        global PR_time
        PR_time = 0
        probBox.configure(text='0')
        PRBox.configure(text='0')

    def calculate_angle(a, b, c):
        """Calculate the angle between three points."""
        a = np.array(a) # First
        b = np.array(b) # Mid
        c = np.array(c) # End
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return round(angle, 2)

    def update_table():
        """Update the table with the series data and store it in an Excel file."""
        global series_data, table
        table.delete('1.0', tk.END)  # Clear the table
        
        # Create a DataFrame from the series_data
        df = pd.DataFrame(series_data, columns=['Series', 'Repetitions'])
        
        # Update the table in the GUI
        for series, reps in series_data:
            table.insert(tk.END, f"Serie {series}: {reps} reps\n")

    def on_double_click(event):
        """Handle double-click event to modify reps."""
        global series_data, table
        # Get the line number where the double-click occurred
        line_index = table.index("@%s,%s" % (event.x, event.y)).split('.')[0]
        line_index = int(line_index) - 1  # Convert to zero-based index
        
        if 0 <= line_index < len(series_data):
            # Prompt the user to enter a new value for reps
            new_reps = simpledialog.askinteger("Input", f"Enter new repetitions for Serie {series_data[line_index][0]}:", minvalue=0)
            if new_reps is not None:
                # Update the series_data with the new value
                series_data[line_index] = (series_data[line_index][0], new_reps)
                # Update the table
                update_table()

    def store_series(counterBox):
        """Store the current counter value as a series and reset the counter."""
        global series_data, counter, current_series
        reps = int(counterBox.cget("text"))
        series_data.append((current_series, reps))
        update_table()
        current_series += 1
        reset_counter()
        counterBox.configure(text='0')

    def update_frame(cap, pose, lmain, counterBox, probBox, PRBox,table):
        """Update the frame for the video capture and UI."""
        ret, frame = cap.read()
        
        if not ret:
            lmain.after(10, update_frame, cap, pose, lmain, counterBox, probBox, PRBox,table)
            return
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
    
        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            
            # Calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)
            angle2 = calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            # Visualize angle
            cv2.putText(image, str(angle), 
                        tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA) # emplacement du texte: coude + 10 pixels à droite et 10 pixels en bas
            
            # Condition pour +1 HSPU
            global stage, counter, timer_started, start_time, PR_time
            if (ankle[1] < hip[1] or right_ankle[1] < right_hip[1])  and (ankle[1] < wrist[1] or right_ankle[1] < right_wrist[1]):
                if angle > 160:
                    if stage == "up":
                        counter += 1
                    stage = "down"
                if (angle < 100 or angle2 < 100) and stage == 'down':
                    stage = "up"
                    #counter += 1
                    print(counter)
            
            # Condition pour démarrer le chrono pour le handstand
            if (ankle[1] < hip[1] or right_ankle[1] < right_hip[1])and (ankle[1] < wrist[1] or right_ankle[1] < right_wrist[1]):
                if not timer_started:
                    timer_started = True
                    start_time = time.time()
            else:
                timer_started = False
            
            # Affichage du chrono
            if timer_started:
                elapsed_time = time.time() - start_time  # Temps écoulé = temps actuel - temps de départ
                probBox.configure(text=f'{elapsed_time:.2f} s')  # Affichage du temps écoulé avec 2 chiffres après la virgule
                if elapsed_time > PR_time:
                    PR_time = elapsed_time
                PRBox.configure(text=f'{PR_time:.2f} s')  # Affichage du temps écoulé avec 2 chiffres après la virgule
            else:
                probBox.configure(text='0')
                PRBox.configure(text=f'{PR_time:.2f} s')
                    
        except:
            pass
        
        # Affichage du compteur de reps
        # Setup status box
        cv2.rectangle(image, (0, 0), (225, 73), (0, 125, 255), -1) # Rectangle pour le compteur de reps argument 1: image, argument 2: coordonnées du coin supérieur gauche, argument 3: coordonnées du coin inférieur droit, argument 4: couleur, argument 5: épaisseur de la ligne (-1 pour remplir le rectangle)
        
        # Rep data
        cv2.putText(image, 'REPS', (15, 12),  # Affichage du texte "REPS" dans le rectangle
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA) # argument 1: image, argument 2: texte, argument 3: coordonnées du texte, argument 4: police, argument 5: taille de la police, argument 6: couleur, argument 7: épaisseur de la ligne, argument 8: type de ligne
        cv2.putText(image, str(counter), 
                    (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Stage data
        cv2.putText(image, 'STAGE', (65, 12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, 
                    (60, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))               
        
        # Update the image in the Tkinter label
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img) # Convertir l'image pour Tkinter
        lmain.imgtk = imgtk # Mise à jour de l'image
        lmain.configure(image=imgtk)
        
        # Update the counter display
        counterBox.configure(text=str(counter))
        
        # Call this function again after 10 ms
        lmain.after(10, update_frame, cap, pose, lmain, counterBox, probBox, PRBox,table)

    def main():
        """Main function to run the pose detection and UI update loop."""
        window, lmain, counterBox, probBox, PRBox,table = setup_ui()

        cap = cv2.VideoCapture(camera_index)

        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        update_frame(cap, pose, lmain, counterBox, probBox, PRBox,table)

        window.mainloop()

    if __name__ == "__main__":
        main()
    pass

def code_gainage():
    # Initialize MediaPipe pose and drawing utilities
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Initialize global variables
    global counter, stage, PR_rep, series_data, current_series, timer_started, start_time
    counter = 0
    stage = None
    PR_rep = 0
    series_data = []  # List to store series and their repetitions
    current_series = 1
    timer_started = False
    start_time = 0

    class ColorProgressBar(ck.CTkProgressBar): # classe pour la jauge d'amplitude
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set(1)

        def set_value(self, value, color):
            self.set(value)
            self.configure(progress_color=color)

    def setup_ui():
        """Setup the main application window and UI elements."""
        global table
        window = tk.Toplevel()
        window.geometry("1280x720")
        window.title("Plank timer with series table")
        ck.set_appearance_mode("dark")

        # UI Elements
        classLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10, text='Statistics : ')
        classLabel.place(x=10, y=41)

        counterLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10, text='Plank hold')
        counterLabel.place(x=220, y=1)

        PRLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10, text='PR plank')
        PRLabel.place(x=160, y=580)

        counterBox = ck.CTkLabel(window, height=40, width=300, font=("Arial", 20), text_color="white", fg_color="#ff7b00", text='0')
        counterBox.place(x=160, y=41)

        PRBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="Green", text='0')
        PRBox.place(x=160, y=620)

        resetButton = ck.CTkButton(window, text='RESET', command=reset_counter, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="red")
        resetButton.place(x=10, y=620)

        prResetButton = ck.CTkButton(window, text='RESET PR', command=lambda: reset_pr_time(PRBox), height=40, width=120, font=("Arial", 20), text_color="white", fg_color="red")
        prResetButton.place(x=300, y=620)

        # Button to store reps in the series table
        setButton = ck.CTkButton(window, text='SET', command=lambda: store_series(counterBox), height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
        setButton.place(x=450, y=620)

        # Table for series on the right
        tableFrame = tk.Frame(window, bg="black", width=300, height=600)
        tableFrame.place(x=900, y=10)
        tableLabel = ck.CTkLabel(tableFrame, text="Séries et hold", font=("Arial", 16), text_color="white")
        tableLabel.pack()

        table = tk.Text(tableFrame, width=30, height=35, bg="white", fg="black", font=("Arial", 12))
        table.pack()

        frame = tk.Frame(window,height=480, width=480)
        frame.place(x=10, y=90)
        lmain = tk.Label(frame)
        lmain.place(x=0, y=0)

        frame = tk.Frame(window,height=480, width=480)
        frame.place(x=10, y=90)
        lmain = tk.Label(frame)
        lmain.place(x=0, y=0)

        # Progress bars for alignment
        knee_foot_hip_bar = ColorProgressBar(window, height=20, width=120)
        knee_foot_hip_bar.place(x=500, y=100)
        knee_foot_hip_label = ck.CTkLabel(window, height=20, width=120, font=("Arial", 12), text_color="black", text="feet-knee-hip")
        knee_foot_hip_label.place(x=500, y=80)

        knee_hip_shoulder_bar = ColorProgressBar(window, height=20, width=120)
        knee_hip_shoulder_bar.place(x=500, y=160)
        knee_hip_shoulder_label = ck.CTkLabel(window, height=20, width=120, font=("Arial", 12), text_color="black", text="knee-hip-shoulder")
        knee_hip_shoulder_label.place(x=500, y=140)
        # Bind the double-click event to the table
        table.bind("<Double-1>", on_double_click)
        return window, lmain, counterBox, PRBox, knee_foot_hip_bar, knee_hip_shoulder_bar,table

    def reset_counter():
        """Reset the counter and stage to zero."""
        global counter, stage
        counter = 0
        stage = None

    def reset_pr_time(PRBox):
        """Reset the PR rep and update the display."""
        global PR_rep
        PR_rep = 0
        PRBox.configure(text='0')

    def calculate_angle(a, b, c):
        """Calculate the angle between three points."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return round(angle, 2)

    def update_table():
        """Update the table with the series data and store it in an Excel file."""
        global series_data, table
        table.delete('1.0', tk.END)  # Clear the table
        
        # Create a DataFrame from the series_data
        df = pd.DataFrame(series_data, columns=['Series', 'Repetitions'])
        
        # Update the table in the GUI
        for series, reps in series_data:
            table.insert(tk.END, f"Serie {series}: {reps} reps\n")

    def on_double_click(event):
        """Handle double-click event to modify reps."""
        global series_data, table
        # Get the line number where the double-click occurred
        line_index = table.index("@%s,%s" % (event.x, event.y)).split('.')[0]
        line_index = int(line_index) - 1  # Convert to zero-based index
        
        if 0 <= line_index < len(series_data):
            # Prompt the user to enter a new value for reps
            new_reps = simpledialog.askinteger("Input", f"Enter new repetitions for Serie {series_data[line_index][0]}:", minvalue=0)
            if new_reps is not None:
                # Update the series_data with the new value
                series_data[line_index] = (series_data[line_index][0], new_reps)
                # Update the table
                update_table()

    def store_series(counterBox):
        """Store the current counter value as a series and reset the counter."""
        global series_data, counter, current_series
        reps = int(counterBox.cget("text"))
        series_data.append((current_series, reps))
        update_table()
        current_series += 1
        reset_counter()
        counterBox.configure(text='0')

    def update_frame(cap, pose, lmain, counterBox, PRBox, knee_foot_hip_bar, knee_hip_shoulder_bar,table):
        """Update the frame for the video capture and UI."""
        global counter, stage, PR_rep

        ret, frame = cap.read()
        if not ret:
            lmain.after(10, update_frame, cap, pose, lmain, counterBox, PRBox, knee_foot_hip_bar, knee_hip_shoulder_bar,table)
            return

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_foot = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                right_foot = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                knee_foot_hip_angle = calculate_angle(left_foot, left_knee, left_hip)
                knee_hip_shoulder_angle = calculate_angle(left_knee, left_hip, left_shoulder)

            # Condition pour démarrer le chrono pour le gainage
            global timer_started, start_time, PR_rep
            if (left_wrist[0] < left_elbow[0] or right_wrist[0] < right_elbow[0]) and knee_hip_shoulder_angle > 150 and knee_foot_hip_angle > 150 and (left_foot[0] > left_knee[0] > left_hip[0] > left_shoulder[0] or right_foot[0] > right_knee[0] > right_hip[0] > right_shoulder[0]) and (0 < left_foot[0] < 1) and (0 < right_foot[1] < 1):
                if not timer_started:
                    timer_started = True
                    start_time = time.time()
            else:
                timer_started = False
            
            # Affichage du chrono
            if timer_started:
                counter = round(time.time() - start_time, 1)  # Temps écoulé = temps actuel - temps de départ
                counterBox.configure(text=f'{counter:.2f} s')  # Affichage du temps écoulé avec 2 chiffres après la virgule
                if counter > PR_rep:
                    PR_rep = counter
                PRBox.configure(text=f'{PR_rep} s')  # Affichage du temps écoulé avec 2 chiffres après la virgule
            else:
                if counter > 3:
                    store_series(counterBox)
                counterBox.configure(text='0')
                PRBox.configure(text=f'{PR_rep} s')
                    

            # Draw amplitude gauge
            gauge_value = abs((knee_foot_hip_angle)+(knee_hip_shoulder_angle)) / (320)
            gauge_value = np.clip(gauge_value, 0, 1)
            gauge_height = int(200 * gauge_value)

            cv2.rectangle(image, (20, 300), (60, 300 - gauge_height), (0, 255, 0), -1)
            cv2.rectangle(image, (20, 100), (60, 300), (255, 255, 255), 2)
            cv2.putText(image, "% Alignement", (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Update alignment bars
            knee_foot_hip_color = "#00FF00" if 160 <= knee_foot_hip_angle <= 180 else "#FF0000"
            knee_hip_shoulder_color = "#00FF00" if 160 <= knee_hip_shoulder_angle <= 180 else "#FF0000"

            knee_foot_hip_bar.set_value((knee_foot_hip_angle - 160) / 20, knee_foot_hip_color)
            knee_hip_shoulder_bar.set_value((knee_hip_shoulder_angle - 160) / 20, knee_hip_shoulder_color)

            # Draw connections with color based on alignment
            connections_color = (0, 255, 0) if knee_foot_hip_color == "#00FF00" and knee_hip_shoulder_color == "#00FF00" else (255, 0, 0)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=connections_color, thickness=2, circle_radius=2),
                                        mp_drawing.DrawingSpec(color=connections_color, thickness=2, circle_radius=2))

        except Exception as e:
            print(e)

        # Display reps and stage
        cv2.rectangle(image, (0, 0), (225, 73), (0, 125, 255), -1)
        cv2.putText(image, 'Hold', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) 

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)

        counterBox.configure(text=str(counter))

        lmain.after(10, update_frame, cap, pose, lmain, counterBox, PRBox, knee_foot_hip_bar, knee_hip_shoulder_bar,table)

    def main():
        """Main function to run the pose detection and UI update loop."""
        window, lmain, counterBox, PRBox, knee_foot_hip_bar, knee_hip_shoulder_bar,table = setup_ui()

        cap = cv2.VideoCapture(camera_index)
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            update_frame(cap, pose, lmain, counterBox, PRBox, knee_foot_hip_bar, knee_hip_shoulder_bar,table)
            window.mainloop()

        cap.release()
        cv2.destroyAllWindows()

    if __name__ == "__main__":
        main()

def code_lsit():
    # Initialize MediaPipe pose and drawing utilities
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Initialize global variables
    global counter, stage, PR_rep, series_data, current_series, timer_started, start_time
    counter = 0
    stage = None
    PR_rep = 0
    series_data = []  # List to store series and their repetitions
    current_series = 1
    timer_started = False
    start_time = 0

    class ColorProgressBar(ck.CTkProgressBar): # classe pour la jauge d'amplitude
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set(1)

        def set_value(self, value, color):
            self.set(value)
            self.configure(progress_color=color)

    def setup_ui():
        """Setup the main application window and UI elements."""
        global table
        window = tk.Toplevel()
        window.geometry("1280x720")
        window.title("L-sit timer")
        ck.set_appearance_mode("dark")

        # UI Elements
        classLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10, text='Statistics : ')
        classLabel.place(x=10, y=41)

        counterLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10, text='L-sit hold')
        counterLabel.place(x=220, y=1)

        PRLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10, text='PR L-sit')
        PRLabel.place(x=160, y=580)

        counterBox = ck.CTkLabel(window, height=40, width=300, font=("Arial", 20), text_color="white", fg_color="#ff7b00", text='0')
        counterBox.place(x=160, y=41)

        PRBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="Green", text='0')
        PRBox.place(x=160, y=620)

        resetButton = ck.CTkButton(window, text='RESET', command=reset_counter, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="red")
        resetButton.place(x=10, y=620)

        prResetButton = ck.CTkButton(window, text='RESET PR', command=lambda: reset_pr_time(PRBox), height=40, width=120, font=("Arial", 20), text_color="white", fg_color="red")
        prResetButton.place(x=300, y=620)

        # Button to store reps in the series table
        setButton = ck.CTkButton(window, text='SET', command=lambda: store_series(counterBox), height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
        setButton.place(x=450, y=620)

        # Table for series on the right
        tableFrame = tk.Frame(window, bg="black", width=300, height=600)
        tableFrame.place(x=900, y=10)
        tableLabel = ck.CTkLabel(tableFrame, text="Series and hold", font=("Arial", 16), text_color="white")
        tableLabel.pack()

        table = tk.Text(tableFrame, width=30, height=35, bg="white", fg="black", font=("Arial", 12))
        table.pack()

        frame = tk.Frame(window,height=480, width=480)
        frame.place(x=10, y=90)
        lmain = tk.Label(frame)
        lmain.place(x=0, y=0)

        frame = tk.Frame(window,height=480, width=480)
        frame.place(x=10, y=90)
        lmain = tk.Label(frame)
        lmain.place(x=0, y=0)

        # Progress bars for alignment
        knee_foot_hip_bar = ColorProgressBar(window, height=20, width=120)
        knee_foot_hip_bar.place(x=500, y=100)
        knee_foot_hip_label = ck.CTkLabel(window, height=20, width=120, font=("Arial", 12), text_color="black", text="feet-knee-hip")
        knee_foot_hip_label.place(x=500, y=80)

        hip_shoulder_foot_bar = ColorProgressBar(window, height=20, width=120)
        hip_shoulder_foot_bar.place(x=500, y=160)
        hip_shoulder_foot_label = ck.CTkLabel(window, height=20, width=120, font=("Arial", 12), text_color="black", text="shoulder-hip-feet")
        hip_shoulder_foot_label.place(x=500, y=140)

        # Bind the double-click event to the table
        table.bind("<Double-1>", on_double_click)

        return window, lmain, counterBox, PRBox, knee_foot_hip_bar, hip_shoulder_foot_bar,table

    def reset_counter():
        """Reset the counter and stage to zero."""
        global counter, stage
        counter = 0
        stage = None

    def reset_pr_time(PRBox):
        """Reset the PR rep and update the display."""
        global PR_rep
        PR_rep = 0
        PRBox.configure(text='0')

    def calculate_angle(a, b, c):
        """Calculate the angle between three points."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return round(angle, 2)

    def update_table():
        """Update the table with the series data and store it in an Excel file."""
        global series_data, table
        table.delete('1.0', tk.END)  # Clear the table
        
        # Create a DataFrame from the series_data
        df = pd.DataFrame(series_data, columns=['Series', 'Repetitions'])
        
        # Update the table in the GUI
        for series, reps in series_data:
            table.insert(tk.END, f"Serie {series}: {reps} reps\n")

    def on_double_click(event):
        """Handle double-click event to modify reps."""
        global series_data, table
        # Get the line number where the double-click occurred
        line_index = table.index("@%s,%s" % (event.x, event.y)).split('.')[0]
        line_index = int(line_index) - 1  # Convert to zero-based index
        
        if 0 <= line_index < len(series_data):
            # Prompt the user to enter a new value for reps
            new_reps = simpledialog.askinteger("Input", f"Enter new repetitions for Serie {series_data[line_index][0]}:", minvalue=0)
            if new_reps is not None:
                # Update the series_data with the new value
                series_data[line_index] = (series_data[line_index][0], new_reps)
                # Update the table
                update_table()

    def store_series(counterBox):
        """Store the current counter value as a series and reset the counter."""
        global series_data, counter, current_series
        reps = int(counterBox.cget("text"))
        series_data.append((current_series, reps))
        update_table()
        current_series += 1
        reset_counter()
        counterBox.configure(text='0')

    def update_frame(cap, pose, lmain, counterBox, PRBox, knee_foot_hip_bar, hip_shoulder_foot_bar,table):
        """Update the frame for the video capture and UI."""
        global counter, stage, PR_rep

        ret, frame = cap.read()
        if not ret:
            lmain.after(10, update_frame, cap, pose, lmain, counterBox, PRBox, knee_foot_hip_bar, hip_shoulder_foot_bar,table)
            return

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_foot = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_foot = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                knee_foot_hip_angle = calculate_angle(left_foot, left_knee, left_hip)
                hip_shoulder_foot_angle = calculate_angle(left_shoulder, left_hip, left_foot)

            # Condition pour démarrer le chrono pour le l-sit
            global timer_started, start_time, PR_rep
            if (left_wrist[1] > left_elbow[1] or right_wrist[1] > right_elbow[1]) and hip_shoulder_foot_angle < 100  and knee_foot_hip_angle > 150 and (0 < left_foot[0] < 1) and (0 < right_foot[1] < 1):
                if not timer_started:
                    timer_started = True
                    start_time = time.time()
            else:
                timer_started = False
            
            # Affichage du chrono
            if timer_started:
                counter = round(time.time() - start_time, 1)  # Temps écoulé = temps actuel - temps de départ
                counterBox.configure(text=f'{counter:.2f} s')  # Affichage du temps écoulé avec 2 chiffres après la virgule
                if counter > PR_rep:
                    PR_rep = counter
                PRBox.configure(text=f'{PR_rep} s')  # Affichage du temps écoulé avec 2 chiffres après la virgule
            else:
                if counter > 2: # Si le temps de l-sit est superieur à 2 secondes, on prend la série -> avant maj : counter != 0
                    store_series(counterBox)
                counterBox.configure(text='0')
                PRBox.configure(text=f'{PR_rep} s')
                    

            # Draw amplitude gauge
            gauge_value = abs(knee_foot_hip_angle+hip_shoulder_foot_angle) / (320)
            gauge_value = np.clip(gauge_value, 0, 1)
            gauge_height = int(200 * gauge_value)

            cv2.rectangle(image, (20, 300), (60, 300 - gauge_height), (0, 255, 0), -1)
            cv2.rectangle(image, (20, 100), (60, 300), (255, 255, 255), 2)
            cv2.putText(image, "% Alignement", (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Update alignment bars
            knee_foot_hip_color = "#00FF00" if 160 <= knee_foot_hip_angle <= 180 else "#FF0000"
            hip_shoulder_foot_color = "#00FF00" if 0 <= hip_shoulder_foot_angle <= 90 else "#FF0000"

            knee_foot_hip_bar.set_value((knee_foot_hip_angle - 160) / 20, knee_foot_hip_color)
            hip_shoulder_foot_bar.set_value((hip_shoulder_foot_angle - 160) / 20, hip_shoulder_foot_color)

            # Draw connections with color based on alignment
            connections_color = (0, 255, 0) if knee_foot_hip_color == "#00FF00" and hip_shoulder_foot_color == "#00FF00" else  (255, 0, 0)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=connections_color, thickness=2, circle_radius=2),
                                        mp_drawing.DrawingSpec(color=connections_color, thickness=2, circle_radius=2))

        except Exception as e:
            print(e)

        # Display reps and stage
        cv2.rectangle(image, (0, 0), (225, 73), (0, 125, 255), -1)
        cv2.putText(image, 'Hold', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) 

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)

        counterBox.configure(text=str(counter))

        lmain.after(10, update_frame, cap, pose, lmain, counterBox, PRBox, knee_foot_hip_bar, hip_shoulder_foot_bar,table)

    def main():
        """Main function to run the pose detection and UI update loop."""
        window, lmain, counterBox, PRBox, knee_foot_hip_bar, hip_shoulder_foot_bar,table = setup_ui()

        cap = cv2.VideoCapture(camera_index)
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            update_frame(cap, pose, lmain, counterBox, PRBox, knee_foot_hip_bar, hip_shoulder_foot_bar,table)
            window.mainloop()

        cap.release()
        cv2.destroyAllWindows()

    if __name__ == "__main__":
        main()

def code_frontlever():
    # Initialize MediaPipe pose and drawing utilities
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Initialize global variables
    global counter, stage, PR_rep, series_data, current_series, timer_started, start_time
    counter = 0
    stage = None
    PR_rep = 0
    series_data = []  # List to store series and their repetitions
    current_series = 1
    timer_started = False
    start_time = 0

    class ColorProgressBar(ck.CTkProgressBar): # classe pour la jauge d'amplitude
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set(1)

        def set_value(self, value, color):
            self.set(value)
            self.configure(progress_color=color)

    def setup_ui():
        """Setup the main application window and UI elements."""
        global table
        window = tk.Toplevel()
        window.geometry("1280x720")
        window.title("Front-lever timer with series table")
        ck.set_appearance_mode("dark")

        # UI Elements
        classLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10, text='Statistics : ')
        classLabel.place(x=10, y=41)

        counterLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10, text='Front-lever hold')
        counterLabel.place(x=220, y=1)

        PRLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10, text='PR front-lever')
        PRLabel.place(x=160, y=580)

        counterBox = ck.CTkLabel(window, height=40, width=300, font=("Arial", 20), text_color="white", fg_color="#ff7b00", text='0')
        counterBox.place(x=160, y=41)

        PRBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="Green", text='0')
        PRBox.place(x=160, y=620)

        resetButton = ck.CTkButton(window, text='RESET', command=reset_counter, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="red")
        resetButton.place(x=10, y=620)

        prResetButton = ck.CTkButton(window, text='RESET PR', command=lambda: reset_pr_time(PRBox), height=40, width=120, font=("Arial", 20), text_color="white", fg_color="red")
        prResetButton.place(x=300, y=620)

        # Button to store reps in the series table
        setButton = ck.CTkButton(window, text='SET', command=lambda: store_series(counterBox), height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
        setButton.place(x=450, y=620)

        # Table for series on the right
        tableFrame = tk.Frame(window, bg="black", width=300, height=600)
        tableFrame.place(x=900, y=10)
        tableLabel = ck.CTkLabel(tableFrame, text="Series and hold", font=("Arial", 16), text_color="white")
        tableLabel.pack()

        table = tk.Text(tableFrame, width=30, height=35, bg="white", fg="black", font=("Arial", 12))
        table.pack()

        frame = tk.Frame(window,height=480, width=480)
        frame.place(x=10, y=90)
        lmain = tk.Label(frame)
        lmain.place(x=0, y=0)

        frame = tk.Frame(window,height=480, width=480)
        frame.place(x=10, y=90)
        lmain = tk.Label(frame)
        lmain.place(x=0, y=0)

        # Progress bars for alignment
        knee_foot_hip_bar = ColorProgressBar(window, height=20, width=120)
        knee_foot_hip_bar.place(x=500, y=100)
        knee_foot_hip_label = ck.CTkLabel(window, height=20, width=120, font=("Arial", 12), text_color="black", text="feet-knee-hip")
        knee_foot_hip_label.place(x=500, y=80)

        knee_hip_shoulder_bar = ColorProgressBar(window, height=20, width=120)
        knee_hip_shoulder_bar.place(x=500, y=160)
        knee_hip_shoulder_label = ck.CTkLabel(window, height=20, width=120, font=("Arial", 12), text_color="black", text="knee-hip-shoulder")
        knee_hip_shoulder_label.place(x=500, y=140)

        # Bind the double-click event to the table
        table.bind("<Double-1>", on_double_click)

        return window, lmain, counterBox, PRBox, knee_foot_hip_bar, knee_hip_shoulder_bar,table

    def reset_counter():
        """Reset the counter and stage to zero."""
        global counter, stage
        counter = 0
        stage = None

    def reset_pr_time(PRBox):
        """Reset the PR rep and update the display."""
        global PR_rep
        PR_rep = 0
        PRBox.configure(text='0')

    def calculate_angle(a, b, c):
        """Calculate the angle between three points."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return round(angle, 2)

    def update_table():
        """Update the table with the series data and store it in an Excel file."""
        global series_data, table
        table.delete('1.0', tk.END)  # Clear the table
        
        # Create a DataFrame from the series_data
        df = pd.DataFrame(series_data, columns=['Series', 'Repetitions'])
        
        # Update the table in the GUI
        for series, reps in series_data:
            table.insert(tk.END, f"Serie {series}: {reps} reps\n")

    def on_double_click(event):
        """Handle double-click event to modify reps."""
        global series_data, table
        # Get the line number where the double-click occurred
        line_index = table.index("@%s,%s" % (event.x, event.y)).split('.')[0]
        line_index = int(line_index) - 1  # Convert to zero-based index
        
        if 0 <= line_index < len(series_data):
            # Prompt the user to enter a new value for reps
            new_reps = simpledialog.askinteger("Input", f"Enter new repetitions for Serie {series_data[line_index][0]}:", minvalue=0)
            if new_reps is not None:
                # Update the series_data with the new value
                series_data[line_index] = (series_data[line_index][0], new_reps)
                # Update the table
                update_table()

    def store_series(counterBox):
        """Store the current counter value as a series and reset the counter."""
        global series_data, counter, current_series
        reps = int(counterBox.cget("text"))
        series_data.append((current_series, reps))
        update_table()
        current_series += 1
        reset_counter()
        counterBox.configure(text='0')

    def update_frame(cap, pose, lmain, counterBox, PRBox, knee_foot_hip_bar, knee_hip_shoulder_bar,table):
        """Update the frame for the video capture and UI."""
        global counter, stage, PR_rep

        ret, frame = cap.read()
        if not ret:
            lmain.after(10, update_frame, cap, pose, lmain, counterBox, PRBox, knee_foot_hip_bar, knee_hip_shoulder_bar,table)
            return

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            left_hand = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            right_hand = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            left_mouth = [landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].x, landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].y]
            right_mouth = [landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].x, landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]


            knee_foot_hip_angle = calculate_angle(left_ankle, left_knee, left_hip)
            knee_hip_shoulder_angle = calculate_angle(left_knee, left_hip, left_shoulder)
            knee_hip_shoulder_angle2 = calculate_angle(right_knee, right_hip, right_shoulder)
            knee_foot_hip_angle2 = calculate_angle(right_ankle, right_knee, right_hip)
            left_shoulder_elbow_wrist_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_shoulder_elbow_wrist_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

########################################################################################################
########################################################################################################
                                    # Section conditions pour compteur
########################################################################################################
########################################################################################################                                
            # Condition pour démarrer le chrono pour le front-lever
            global timer_started, start_time, PR_rep
            if (left_wrist[1] < left_elbow[1] < left_shoulder[1] or right_wrist[1] < right_elbow[1] < right_shoulder[1]) and (knee_hip_shoulder_angle > 155 and knee_foot_hip_angle > 155) or (knee_hip_shoulder_angle2 > 155 and knee_foot_hip_angle2 > 155) > 155 and (left_ankle[0] < left_knee[0] < left_hip[0] < left_shoulder[0] or right_ankle[0] > right_knee[0] > right_hip[0] > right_shoulder[0]) and (0 < left_ankle[0] < 1) and (0 < right_ankle[1] < 1) and (left_shoulder_elbow_wrist_angle >155 or right_shoulder_elbow_wrist_angle > 155) : #Changer conditions d'angles
                if not timer_started:
                    timer_started = True
                    start_time = time.time()
            else:
                timer_started = False
            
            # Affichage du chrono
            if timer_started:
                counter = round(time.time() - start_time, 1)  # Temps écoulé = temps actuel - temps de départ
                counterBox.configure(text=f'{counter:.2f} s')  # Affichage du temps écoulé avec 2 chiffres après la virgule
                if counter > PR_rep:
                    PR_rep = counter
                PRBox.configure(text=f'{PR_rep} s')  # Affichage du temps écoulé avec 2 chiffres après la virgule
            else:
                if counter > 1:
                    store_series(counterBox)
                counterBox.configure(text='0')
                PRBox.configure(text=f'{PR_rep} s')
                    

            # Draw amplitude gauge
            gauge_value = abs((knee_foot_hip_angle)+(knee_hip_shoulder_angle)) / (320)
            gauge_value = np.clip(gauge_value, 0, 1)
            gauge_height = int(200 * gauge_value)

            cv2.rectangle(image, (20, 300), (60, 300 - gauge_height), (0, 255, 0), -1)
            cv2.rectangle(image, (20, 100), (60, 300), (255, 255, 255), 2)
            cv2.putText(image, "% Alignement", (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Update alignment bars
            knee_foot_hip_color = "#00FF00" if 160 <= knee_foot_hip_angle <= 180 else "#FF0000"
            knee_hip_shoulder_color = "#00FF00" if 160 <= knee_hip_shoulder_angle <= 180 else "#FF0000"

            knee_foot_hip_bar.set_value((knee_foot_hip_angle - 160) / 20, knee_foot_hip_color)
            knee_hip_shoulder_bar.set_value((knee_hip_shoulder_angle - 160) / 20, knee_hip_shoulder_color)

            # Draw connections with color based on alignment
            connections_color = (0, 255, 0) if knee_foot_hip_color == "#00FF00" and knee_hip_shoulder_color == "#00FF00" else (255, 0, 0)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=connections_color, thickness=2, circle_radius=2),
                                        mp_drawing.DrawingSpec(color=connections_color, thickness=2, circle_radius=2))

        except Exception as e:
            print(e)

        # Display reps and stage
        cv2.rectangle(image, (0, 0), (225, 73), (0, 125, 255), -1)
        cv2.putText(image, 'Hold', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) 

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)

        counterBox.configure(text=str(counter))

        lmain.after(10, update_frame, cap, pose, lmain, counterBox, PRBox, knee_foot_hip_bar, knee_hip_shoulder_bar,table)

    def main():
        """Main function to run the pose detection and UI update loop."""
        window, lmain, counterBox, PRBox, knee_foot_hip_bar, knee_hip_shoulder_bar,table = setup_ui()

        cap = cv2.VideoCapture(camera_index)
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            update_frame(cap, pose, lmain, counterBox, PRBox, knee_foot_hip_bar, knee_hip_shoulder_bar,table)
            window.mainloop()

        cap.release()
        cv2.destroyAllWindows()

    if __name__ == "__main__":
        main()

def code_planche():
    # Initialize MediaPipe pose and drawing utilities
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Initialize global variables
    global counter, stage, PR_rep, series_data, current_series, timer_started, start_time
    counter = 0
    stage = None
    PR_rep = 0
    series_data = []  # List to store series and their repetitions
    current_series = 1
    timer_started = False
    start_time = 0

    class ColorProgressBar(ck.CTkProgressBar): # classe pour la jauge d'amplitude
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set(1)

        def set_value(self, value, color):
            self.set(value)
            self.configure(progress_color=color)

    def setup_ui():
        """Setup the main application window and UI elements."""
        global table
        window = tk.Toplevel()
        window.geometry("1280x720")
        window.title("Planche timer")
        ck.set_appearance_mode("dark")

        # UI Elements
        classLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10, text='Statistics : ')
        classLabel.place(x=10, y=41)

        counterLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10, text='planche hold')
        counterLabel.place(x=220, y=1)

        PRLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10, text='PR planche')
        PRLabel.place(x=160, y=580)

        counterBox = ck.CTkLabel(window, height=40, width=300, font=("Arial", 20), text_color="white", fg_color="#ff7b00", text='0')
        counterBox.place(x=160, y=41)

        PRBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="Green", text='0')
        PRBox.place(x=160, y=620)

        resetButton = ck.CTkButton(window, text='RESET', command=reset_counter, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="red")
        resetButton.place(x=10, y=620)

        prResetButton = ck.CTkButton(window, text='RESET PR', command=lambda: reset_pr_time(PRBox), height=40, width=120, font=("Arial", 20), text_color="white", fg_color="red")
        prResetButton.place(x=300, y=620)

        # Button to store reps in the series table
        setButton = ck.CTkButton(window, text='SET', command=lambda: store_series(counterBox), height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
        setButton.place(x=450, y=620)

        # Table for series on the right
        tableFrame = tk.Frame(window, bg="black", width=300, height=600)
        tableFrame.place(x=900, y=10)
        tableLabel = ck.CTkLabel(tableFrame, text="Series and hold", font=("Arial", 16), text_color="white")
        tableLabel.pack()

        table = tk.Text(tableFrame, width=30, height=35, bg="white", fg="black", font=("Arial", 12))
        table.pack()

        frame = tk.Frame(window,height=480, width=480)
        frame.place(x=10, y=90)
        lmain = tk.Label(frame)
        lmain.place(x=0, y=0)

        frame = tk.Frame(window,height=480, width=480)
        frame.place(x=10, y=90)
        lmain = tk.Label(frame)
        lmain.place(x=0, y=0)

        # Progress bars for alignment
        knee_foot_hip_bar = ColorProgressBar(window, height=20, width=120)
        knee_foot_hip_bar.place(x=500, y=100)
        knee_foot_hip_label = ck.CTkLabel(window, height=20, width=120, font=("Arial", 12), text_color="black", text="feet-knee-hip")
        knee_foot_hip_label.place(x=500, y=80)

        knee_hip_shoulder_bar = ColorProgressBar(window, height=20, width=120)
        knee_hip_shoulder_bar.place(x=500, y=160)
        knee_hip_shoulder_label = ck.CTkLabel(window, height=20, width=120, font=("Arial", 12), text_color="black", text="knee-hip-shoulder")
        knee_hip_shoulder_label.place(x=500, y=140)

        # Bind the double-click event to the table
        table.bind("<Double-1>", on_double_click)

        return window, lmain, counterBox, PRBox, knee_foot_hip_bar, knee_hip_shoulder_bar,table

    def reset_counter():
        """Reset the counter and stage to zero."""
        global counter, stage
        counter = 0
        stage = None

    def reset_pr_time(PRBox):
        """Reset the PR rep and update the display."""
        global PR_rep
        PR_rep = 0
        PRBox.configure(text='0')

    def calculate_angle(a, b, c):
        """Calculate the angle between three points."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return round(angle, 2)

    def update_table():
        """Update the table with the series data and store it in an Excel file."""
        global series_data, table
        table.delete('1.0', tk.END)  # Clear the table
        
        # Create a DataFrame from the series_data
        df = pd.DataFrame(series_data, columns=['Series', 'Repetitions'])
        
        # Update the table in the GUI
        for series, reps in series_data:
            table.insert(tk.END, f"Série {series}: {reps} reps\n")

    def on_double_click(event):
        """Handle double-click event to modify reps."""
        global series_data, table
        # Get the line number where the double-click occurred
        line_index = table.index("@%s,%s" % (event.x, event.y)).split('.')[0]
        line_index = int(line_index) - 1  # Convert to zero-based index
        
        if 0 <= line_index < len(series_data):
            # Prompt the user to enter a new value for reps
            new_reps = simpledialog.askinteger("Input", f"Enter new repetitions for Serie {series_data[line_index][0]}:", minvalue=0)
            if new_reps is not None:
                # Update the series_data with the new value
                series_data[line_index] = (series_data[line_index][0], new_reps)
                # Update the table
                update_table()

    def store_series(counterBox):
        """Store the current counter value as a series and reset the counter."""
        global series_data, counter, current_series
        reps = int(counterBox.cget("text"))
        series_data.append((current_series, reps))
        update_table()
        current_series += 1
        reset_counter()
        counterBox.configure(text='0')

    def update_frame(cap, pose, lmain, counterBox, PRBox, knee_foot_hip_bar, knee_hip_shoulder_bar,table):
        """Update the frame for the video capture and UI."""
        global counter, stage, PR_rep

        ret, frame = cap.read()
        if not ret:
            lmain.after(10, update_frame, cap, pose, lmain, counterBox, PRBox, knee_foot_hip_bar, knee_hip_shoulder_bar,table)
            return

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                left_hand = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                right_hand = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                left_mouth = [landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].x, landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].y]
                right_mouth = [landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].x, landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                knee_foot_hip_angle = calculate_angle(left_ankle, left_knee, left_hip)
                knee_hip_shoulder_angle = calculate_angle(left_knee, left_hip, left_shoulder)
                knee_hip_shoulder_angle2 = calculate_angle(right_knee, right_hip, right_shoulder)
                knee_foot_hip_angle2 = calculate_angle(right_ankle, right_knee, right_hip)
                shoulder_elbow_wrist_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                shoulder_elbow_wrist_angle2 = calculate_angle(right_shoulder, right_elbow, right_wrist)

            # Condition pour démarrer le chrono pour la planche
            global timer_started, start_time, PR_rep
            if (left_wrist[1] > left_elbow[1] > left_shoulder[1] or right_wrist[1] > right_elbow[1] > right_shoulder[1]) and knee_hip_shoulder_angle > 155 and knee_foot_hip_angle > 155 and (shoulder_elbow_wrist_angle > 155 or shoulder_elbow_wrist_angle2 > 155) and (left_ankle[0] < left_knee[0] < left_hip[0] < left_shoulder[0] or right_ankle[0] > right_knee[0] > right_hip[0] > right_shoulder[0]) and (0 < left_ankle[0] < 1) and (0 < right_ankle[1] < 1) :
                if not timer_started :
                    timer_started = True
                    start_time = time.time()
            else:
                timer_started = False
            
            # Affichage du chrono
            if timer_started:
                counter = round(time.time() - start_time, 1)  # Temps écoulé = temps actuel - temps de départ
                counterBox.configure(text=f'{counter:.2f} s')  # Affichage du temps écoulé avec 2 chiffres après la virgule
                if counter > PR_rep:
                    PR_rep = counter
                PRBox.configure(text=f'{PR_rep} s')  # Affichage du temps écoulé avec 2 chiffres après la virgule
            else:
                if counter > 1:
                    store_series(counterBox)
                counterBox.configure(text='0')
                PRBox.configure(text=f'{PR_rep} s')
                    

            # Draw amplitude gauge
            gauge_value = abs((knee_foot_hip_angle)+(knee_hip_shoulder_angle)) / (320)
            gauge_value = np.clip(gauge_value, 0, 1)
            gauge_height = int(200 * gauge_value)

            cv2.rectangle(image, (20, 300), (60, 300 - gauge_height), (0, 255, 0), -1)
            cv2.rectangle(image, (20, 100), (60, 300), (255, 255, 255), 2)
            cv2.putText(image, "% Alignement", (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Update alignment bars
            knee_foot_hip_color = "#00FF00" if 160 <= knee_foot_hip_angle <= 180 else "#FF0000"
            knee_hip_shoulder_color = "#00FF00" if 160 <= knee_hip_shoulder_angle <= 180 else "#FF0000"

            knee_foot_hip_bar.set_value((knee_foot_hip_angle - 160) / 20, knee_foot_hip_color)
            knee_hip_shoulder_bar.set_value((knee_hip_shoulder_angle - 160) / 20, knee_hip_shoulder_color)

            # Draw connections with color based on alignment
            connections_color = (0, 255, 0) if knee_foot_hip_color == "#00FF00" and knee_hip_shoulder_color == "#00FF00" else (255, 0, 0)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=connections_color, thickness=2, circle_radius=2),
                                        mp_drawing.DrawingSpec(color=connections_color, thickness=2, circle_radius=2))

        except Exception as e:
            print(e)

        # Display reps and stage
        cv2.rectangle(image, (0, 0), (225, 73), (0, 125, 255), -1)
        cv2.putText(image, 'Hold', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) 

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)

        counterBox.configure(text=str(counter))

        lmain.after(10, update_frame, cap, pose, lmain, counterBox, PRBox, knee_foot_hip_bar, knee_hip_shoulder_bar,table)

    def main():
        """Main function to run the pose detection and UI update loop."""
        window, lmain, counterBox, PRBox, knee_foot_hip_bar, knee_hip_shoulder_bar,table = setup_ui()

        cap = cv2.VideoCapture(camera_index)
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            update_frame(cap, pose, lmain, counterBox, PRBox, knee_foot_hip_bar, knee_hip_shoulder_bar,table)
            window.mainloop()

        cap.release()
        cv2.destroyAllWindows()

    if __name__ == "__main__":
        main()

def code_DC():
    # Initialize MediaPipe pose and drawing utilities
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Initialize global variables
    global counter, stage, PR_rep, series_data, current_series
    counter = 0 
    stage = None
    PR_rep = 0
    series_data = []  # List to store series and their repetitions
    current_series = 1

    def setup_ui():
        """Setup the main application window and UI elements."""
        global table
        window = tk.Toplevel()
        window.geometry("1280x720")
        window.title("Bench reps counter with series table")
        ck.set_appearance_mode("dark")

        # UI Elements
        classLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10, text='Statistics : ')
        classLabel.place(x=10, y=41)

        counterLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10, text='REPS')
        counterLabel.place(x=160, y=1)

        PRLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10, text='PR bench')
        PRLabel.place(x=160, y=580)

        counterBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="#ff7b00", text='0')
        counterBox.place(x=160, y=41)

        PRBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="Green", text='0')
        PRBox.place(x=160, y=620)

        resetButton = ck.CTkButton(window, text='RESET', command=reset_counter, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="red")
        resetButton.place(x=10, y=620)

        prResetButton = ck.CTkButton(window, text='RESET PR', command=lambda: reset_pr_time(PRBox), height=40, width=120, font=("Arial", 20), text_color="white", fg_color="red")
        prResetButton.place(x=300, y=620)

        # Button to store reps in the series table
        setButton = ck.CTkButton(window, text='SET', command=lambda: store_series(counterBox), height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
        setButton.place(x=450, y=620)

        # Table for series on the right
        tableFrame = tk.Frame(window, bg="black", width=300, height=600)
        tableFrame.place(x=900, y=10)
        tableLabel = ck.CTkLabel(tableFrame, text="Series et reps", font=("Arial", 16), text_color="white")
        tableLabel.pack()

        table = tk.Text(tableFrame, width=30, height=35, bg="white", fg="black", font=("Arial", 12))
        table.pack()

        frame = tk.Frame(window,height=480, width=480)
        frame.place(x=10, y=90)
        lmain = tk.Label(frame)
        lmain.place(x=0, y=0)

        # Bind the double-click event to the table
        table.bind("<Double-1>", on_double_click)

        return window, lmain, counterBox, PRBox, table

    def reset_counter(): 
        """Reset the counter and timer to zero."""
        global counter
        counter = 0 


    def reset_pr_time(PRBox):
        """Reset the PR rep and update the display."""
        global PR_rep
        PR_rep = 0
        PRBox.configure(text='0')

    def calculate_angle(a, b, c):
        """Calculate the angle between three points."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return round(angle, 2)

    def update_table():
        """Update the table with the series data and store it in an Excel file."""
        global series_data, table
        table.delete('1.0', tk.END)  # Clear the table
        
        # Create a DataFrame from the series_data
        df = pd.DataFrame(series_data, columns=['Series', 'Repetitions'])
        
        # Update the table in the GUI
        for series, reps in series_data:
            table.insert(tk.END, f"Serie {series}: {reps} reps\n")

    def on_double_click(event):
        """Handle double-click event to modify reps."""
        global series_data, table
        # Get the line number where the double-click occurred
        line_index = table.index("@%s,%s" % (event.x, event.y)).split('.')[0]
        line_index = int(line_index) - 1  # Convert to zero-based index
        
        if 0 <= line_index < len(series_data):
            # Prompt the user to enter a new value for reps
            new_reps = simpledialog.askinteger("Input", f"Enter new repetitions for Serie {series_data[line_index][0]}:", minvalue=0)
            if new_reps is not None:
                # Update the series_data with the new value
                series_data[line_index] = (series_data[line_index][0], new_reps)
                # Update the table
                update_table()

    def store_series(counterBox):
        """Store the current counter value as a series and reset the counter."""
        global series_data, counter, current_series
        reps = int(counterBox.cget("text"))
        series_data.append((current_series, reps))
        update_table()
        current_series += 1
        reset_counter()
        counterBox.configure(text='0')

    def update_frame(cap, pose, lmain, counterBox, PRBox,table):
        """Update the frame for the video capture and UI."""
        ret, frame = cap.read()
        
        if not ret:
            lmain.after(10, update_frame, cap, pose, lmain, counterBox, PRBox,table)
            return
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        try:
            landmarks = results.pose_landmarks.landmark
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            left_eye_outer = [landmarks[mp_pose.PoseLandmark.LEFT_EYE_OUTER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_EYE_OUTER.value].y]
            right_eye_outer = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value].y]
            left_foot = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            right_foot = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            
            angle = calculate_angle(shoulder, elbow, wrist)
            angle2 = calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            global stage, counter,PR_rep
            if (0 < left_foot[0] < 1) and (0 < right_foot[1] < 1) :
                if (angle > 160 or angle2 > 160) and (wrist[1] < shoulder[1] and wrist[1] < left_eye_outer[1] or right_wrist[1]<right_shoulder[1] and right_wrist[1] < right_eye_outer[1]) : # Ajouter des conditions pour éliminer les fausses reps
                    if stage == 'up' :
                        counter += 1
                        print(counter)
                    stage = "down"
                if (angle < 90 or angle2<90) and stage == 'down':
                    stage = "up"
            if counter > PR_rep:
                PR_rep = counter
                PRBox.configure(text=str(PR_rep))

            # Draw amplitude gauge
            min_angle, max_angle = 90, 160
            gauge_value = (angle - min_angle) / (max_angle - min_angle)
            gauge_value = np.clip(gauge_value, 0, 1)
            gauge_height = int(200 * gauge_value)

            cv2.rectangle(image, (20, 300), (60, 300 - gauge_height), (0, 255, 0), -1)
            cv2.rectangle(image, (20, 100), (60, 300), (255, 255, 255), 2)
            cv2.putText(image, 'Range of motion', (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
        except:
            pass

        # Display reps and stage
        cv2.rectangle(image, (0, 0), (225, 73), (0, 125, 255), -1)
        cv2.putText(image, 'REPS', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, 'STAGE', (65, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))               

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        
        counterBox.configure(text=str(counter))
        
        lmain.after(10, update_frame, cap, pose, lmain, counterBox, PRBox,table)

    def main():
        """Main function to run the pose detection and UI update loop."""
        window, lmain, counterBox, PRBox, table = setup_ui()

        cap = cv2.VideoCapture(camera_index)
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            update_frame(cap, pose, lmain, counterBox, PRBox,table)
            window.mainloop()

        cap.release()
        cv2.destroyAllWindows()

    if __name__ == "__main__":
        main()

def code_curl():
    # Initialize MediaPipe pose and drawing utilities
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Initialize global variables
    global counter, stage, PR_rep, series_data, current_series
    counter = 0 
    stage = "up"
    PR_rep = 0
    series_data = []  # List to store series and their repetitions
    current_series = 1

    def setup_ui():
        """Setup the main application window and UI elements."""
        global table
        window = tk.Toplevel()
        window.geometry("1280x720")
        window.title("Biceps curls counter with series table")
        ck.set_appearance_mode("dark")

        # UI Elements
        classLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10, text='Statistics : ')
        classLabel.place(x=10, y=41)

        info_label = ck.CTkLabel(window, height=40, width=120, font=("Arial", 10), text_color="black", padx=10, text='Curl both arms in same time/Do not alternate arms not avaible yet')
        info_label.place(x=500, y=10)

        counterLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10, text='REPS')
        counterLabel.place(x=160, y=1)

        PRLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10, text='PR curls')
        PRLabel.place(x=160, y=580)

        counterBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="#ff7b00", text='0')
        counterBox.place(x=160, y=41)

        PRBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="Green", text='0')
        PRBox.place(x=160, y=620)

        resetButton = ck.CTkButton(window, text='RESET', command=reset_counter, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="red")
        resetButton.place(x=10, y=620)

        prResetButton = ck.CTkButton(window, text='RESET PR', command=lambda: reset_pr_time(PRBox), height=40, width=120, font=("Arial", 20), text_color="white", fg_color="red")
        prResetButton.place(x=300, y=620)

        # Button to store reps in the series table
        setButton = ck.CTkButton(window, text='SET', command=lambda: store_series(counterBox), height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
        setButton.place(x=450, y=620)

        # Table for series on the right
        tableFrame = tk.Frame(window, bg="black", width=300, height=600)
        tableFrame.place(x=900, y=10)
        tableLabel = ck.CTkLabel(tableFrame, text="Series et reps", font=("Arial", 16), text_color="white")
        tableLabel.pack()

        table = tk.Text(tableFrame, width=30, height=35, bg="white", fg="black", font=("Arial", 12))
        table.pack()

        frame = tk.Frame(window,height=480, width=480)
        frame.place(x=10, y=90)
        lmain = tk.Label(frame)
        lmain.place(x=0, y=0)

        # Bind the double-click event to the table
        table.bind("<Double-1>", on_double_click)

        return window, lmain, counterBox, PRBox, table

    def reset_counter(): 
        """Reset the counter and timer to zero."""
        global counter
        counter = 0 


    def reset_pr_time(PRBox):
        """Reset the PR rep and update the display."""
        global PR_rep
        PR_rep = 0
        PRBox.configure(text='0')

    def calculate_angle(a, b, c):
        """Calculate the angle between three points."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return round(angle, 2)

    def update_table():
        """Update the table with the series data and store it in an Excel file."""
        global series_data, table
        table.delete('1.0', tk.END)  # Clear the table
        
        # Create a DataFrame from the series_data
        df = pd.DataFrame(series_data, columns=['Series', 'Repetitions'])
        
        # Update the table in the GUI
        for series, reps in series_data:
            table.insert(tk.END, f"Serie {series}: {reps} reps\n")

    def on_double_click(event):
        """Handle double-click event to modify reps."""
        global series_data, table
        # Get the line number where the double-click occurred
        line_index = table.index("@%s,%s" % (event.x, event.y)).split('.')[0]
        line_index = int(line_index) - 1  # Convert to zero-based index
        
        if 0 <= line_index < len(series_data):
            # Prompt the user to enter a new value for reps
            new_reps = simpledialog.askinteger("Input", f"Enter new repetitions for Serie {series_data[line_index][0]}:", minvalue=0)
            if new_reps is not None:
                # Update the series_data with the new value
                series_data[line_index] = (series_data[line_index][0], new_reps)
                # Update the table
                update_table()

    def store_series(counterBox):
        """Store the current counter value as a series and reset the counter."""
        global series_data, counter, current_series
        reps = int(counterBox.cget("text"))
        series_data.append((current_series, reps))
        update_table()
        current_series += 1
        reset_counter()
        counterBox.configure(text='0')

    def update_frame(cap, pose, lmain, counterBox, PRBox,table):
        """Update the frame for the video capture and UI."""
        ret, frame = cap.read()
        
        if not ret:
            lmain.after(10, update_frame, cap, pose, lmain, counterBox, PRBox,table)
            return
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        try:
            landmarks = results.pose_landmarks.landmark
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_foot = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_foot = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            
            angle = calculate_angle(shoulder, elbow, wrist)
            angle2 = calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            global stage, counter,PR_rep
            if  (shoulder[1]<hip[1] or right_shoulder[1] < right_hip[1]) and shoulder[1] < left_foot[1] and (0 < left_foot[0] < 1) and (0 < right_foot[1] < 1):
                if angle > 160 or angle2 > 160:               # Changement de condition ici avec feet_detected
                    stage = "down"
                if (angle < 80 or angle2<80) and stage == 'down':
                    stage = "up"
                    counter += 1
                    print(counter)
            if counter > PR_rep:
                PR_rep = counter
                PRBox.configure(text=str(PR_rep))

            # Draw amplitude gauge
            min_angle, max_angle = 60, 160
            gauge_value = abs(angle - 160) / (max_angle - min_angle)
            gauge_value = np.clip(gauge_value, 0, 1)
            gauge_height = int(200 * gauge_value)

            cv2.rectangle(image, (20, 300), (60, 300 - gauge_height), (0, 255, 0), -1)
            cv2.rectangle(image, (20, 100), (60, 300), (255, 255, 255), 2)
            cv2.putText(image, 'Range of motion', (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
        except:
            pass

        # Display reps and stage
        cv2.rectangle(image, (0, 0), (225, 73), (0, 125, 255), -1)
        cv2.putText(image, 'REPS', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, 'STAGE', (65, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))               

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        
        counterBox.configure(text=str(counter))
        
        lmain.after(10, update_frame, cap, pose, lmain, counterBox, PRBox,table)

    def main():
        """Main function to run the pose detection and UI update loop."""
        window, lmain, counterBox, PRBox, table = setup_ui()

        cap = cv2.VideoCapture(camera_index)
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            update_frame(cap, pose, lmain, counterBox, PRBox,table)
            window.mainloop()

        cap.release()
        cv2.destroyAllWindows()

    if __name__ == "__main__":
        main()

def code_DL():
    # Initialize MediaPipe pose and drawing utilities
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Initialize global variables
    global counter,stage, PR_rep, series_data, current_series
    counter = 0
    stage = None
    PR_rep = 0
    series_data = []  # List to store series and their repetitions
    current_series = 1

    class ColorProgressBar(ck.CTkProgressBar): # classe pour la jauge d'amplitude
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set(1)

        def set_value(self, value, color):
            self.set(value)
            self.configure(progress_color=color)

    def setup_ui():
        """Setup the main application window and UI elements."""
        global table
        window = tk.Toplevel()
        window.geometry("1280x720")
        window.title("Deadlift Reps Counter")
        ck.set_appearance_mode("dark")

        # UI Elements
        classLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10, text='Statistics : ')
        classLabel.place(x=10, y=41)

        counterLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10, text='REPS')
        counterLabel.place(x=160, y=1)

        PRLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10, text='PR deadlift')
        PRLabel.place(x=160, y=580)

        counterBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="#ff7b00", text='0')
        counterBox.place(x=160, y=41)

        PRBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="Green", text='0')
        PRBox.place(x=160, y=620)

        resetButton = ck.CTkButton(window, text='RESET', command=reset_counter, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="red")
        resetButton.place(x=10, y=620)

        prResetButton = ck.CTkButton(window, text='RESET PR', command=lambda: reset_pr_time(PRBox), height=40, width=120, font=("Arial", 20), text_color="white", fg_color="red")
        prResetButton.place(x=300, y=620)

        attention_label = ck.CTkLabel(window, height=40, width=120, font=("Arial", 10), text_color="black", padx=10, text='Ne pas mettre la caméra complétement de face ni de côté, mettre la caméra partiellement de côté')
        attention_label.place(x=400, y=10)

        attention_label2 = ck.CTkLabel(window, height=40, width=120, font=("Arial", 10), text_color="black", padx=10, text='Do not put the camera completely front or sideways, put the camera partially sideways')
        attention_label2.place(x=400, y=40)

        # Button to store reps in the series table
        setButton = ck.CTkButton(window, text='SET', command=lambda: store_series(counterBox), height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
        setButton.place(x=450, y=620)

        # Table for series on the right
        tableFrame = tk.Frame(window, bg="black", width=300, height=600)
        tableFrame.place(x=900, y=10)
        tableLabel = ck.CTkLabel(tableFrame, text="Series and reps", font=("Arial", 16), text_color="white")
        tableLabel.pack()

        table = tk.Text(tableFrame, width=30, height=35, bg="white", fg="black", font=("Arial", 12))
        table.pack()

        frame = tk.Frame(window,height=480, width=480)
        frame.place(x=10, y=90)
        lmain = tk.Label(frame)
        lmain.place(x=0, y=0)

        frame = tk.Frame(window,height=480, width=480)
        frame.place(x=10, y=90)
        lmain = tk.Label(frame)
        lmain.place(x=0, y=0)

        # Progress bars for alignment
        knee_foot_hip_bar = ColorProgressBar(window, height=20, width=120)
        knee_foot_hip_bar.place(x=500, y=100)
        knee_foot_hip_label = ck.CTkLabel(window, height=20, width=120, font=("Arial", 12), text_color="black", text="feet-knee-hip")
        knee_foot_hip_label.place(x=500, y=80)

        knee_hip_shoulder_bar = ColorProgressBar(window, height=20, width=120)
        knee_hip_shoulder_bar.place(x=500, y=160)
        knee_hip_shoulder_label = ck.CTkLabel(window, height=20, width=120, font=("Arial", 12), text_color="black", text="knee-hip-shoulder")
        knee_hip_shoulder_label.place(x=500, y=140)

        # Bind the double-click event to the table
        table.bind("<Double-1>", on_double_click)

        return window, lmain, counterBox, PRBox, knee_foot_hip_bar, knee_hip_shoulder_bar,table

    def reset_counter():
        """Reset the counter and stage to zero."""
        global counter, stage
        counter = 0
        stage = None

    def reset_pr_time(PRBox):
        """Reset the PR rep and update the display."""
        global PR_rep
        PR_rep = 0
        PRBox.configure(text='0')

    def calculate_angle(a, b, c):
        """Calculate the angle between three points."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return round(angle, 2)

    def update_table():
        """Update the table with the series data and store it in an Excel file."""
        global series_data, table
        table.delete('1.0', tk.END)  # Clear the table
        
        # Create a DataFrame from the series_data
        df = pd.DataFrame(series_data, columns=['Series', 'Repetitions'])
        
        # Update the table in the GUI
        for series, reps in series_data:
            table.insert(tk.END, f"Série {series}: {reps} reps\n")

    def on_double_click(event):
        """Handle double-click event to modify reps."""
        global series_data, table
        # Get the line number where the double-click occurred
        line_index = table.index("@%s,%s" % (event.x, event.y)).split('.')[0]
        line_index = int(line_index) - 1  # Convert to zero-based index
        
        if 0 <= line_index < len(series_data):
            # Prompt the user to enter a new value for reps
            new_reps = simpledialog.askinteger("Input", f"Enter new repetitions for Serie {series_data[line_index][0]}:", minvalue=0)
            if new_reps is not None:
                # Update the series_data with the new value
                series_data[line_index] = (series_data[line_index][0], new_reps)
                # Update the table
                update_table()

    def store_series(counterBox):
        """Store the current counter value as a series and reset the counter."""
        global series_data, counter, current_series, stage
        reps = int(counterBox.cget("text"))
        series_data.append((current_series, reps))
        update_table()
        current_series += 1
        reset_counter()
        counterBox.configure(text='0')
        stage = None

    def update_frame(cap, pose, lmain, counterBox, PRBox, knee_foot_hip_bar, knee_hip_shoulder_bar,table):
        """Update the frame for the video capture and UI."""
        global counter, stage, PR_rep

        ret, frame = cap.read()
        if not ret:
            lmain.after(10, update_frame, cap, pose, lmain, counterBox, PRBox, knee_foot_hip_bar, knee_hip_shoulder_bar,table)
            return

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_foot = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_foot = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                angle = calculate_angle(left_shoulder, left_hip, left_knee)
                angle2 = calculate_angle(right_shoulder, right_hip, right_knee)
                knee_foot_hip_angle = calculate_angle(left_foot, left_knee, left_hip)
                knee_hip_shoulder_angle = calculate_angle(left_knee, left_hip, left_shoulder)

                # Update stage and counter
                if stage == None and left_wrist[1] > left_knee[1] > left_shoulder[1] and right_wrist[1] > right_knee[1] > right_shoulder[1] and (0 < left_foot[0] < 1) and (0 < right_foot[1] < 1) :
                    stage = "down"
                if left_wrist[1] > left_knee[1] > left_shoulder[1] and right_wrist[1] > right_knee[1] > right_shoulder[1] and stage == 'up' and (0 < left_foot[0] < 1) and (0 < right_foot[1] < 1):
                    stage = "down"
                    counter += 1
                    counterBox.configure(text=str(counter))
                if (angle > 160 or angle2 > 160) and stage == 'down':
                    stage = "up"
                if counter > PR_rep:
                    PR_rep = counter
                    PRBox.configure(text=str(PR_rep))

                # Draw amplitude gauge
                min_angle, max_angle = 90, 160
                gauge_value = (angle - min_angle) / (max_angle - min_angle)
                gauge_value = np.clip(gauge_value, 0, 1)
                gauge_height = int(200 * gauge_value)

                cv2.rectangle(image, (20, 300), (60, 300 - gauge_height), (0, 255, 0), -1)
                cv2.rectangle(image, (20, 100), (60, 300), (255, 255, 255), 2)
                cv2.putText(image, 'Range of motion', (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Update alignment bars
                knee_foot_hip_color = "#00FF00" if 160 <= knee_foot_hip_angle <= 220 else "#FF0000"
                knee_hip_shoulder_color = "#00FF00" if 160 <= knee_hip_shoulder_angle <= 220 else "#FF0000"

                knee_foot_hip_bar.set_value((knee_foot_hip_angle - 160) / 20, knee_foot_hip_color)
                knee_hip_shoulder_bar.set_value((knee_hip_shoulder_angle - 160) / 20, knee_hip_shoulder_color)

                # Draw connections with color based on alignment
                connections_color = (0, 255, 0) if knee_foot_hip_color == "#00FF00" and knee_hip_shoulder_color == "#00FF00" else (255, 0, 0)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=connections_color, thickness=2, circle_radius=2),
                                        mp_drawing.DrawingSpec(color=connections_color, thickness=2, circle_radius=2))

        except Exception as e:
            print(e)

        # Display reps and stage
        cv2.rectangle(image, (0, 0), (225, 73), (0, 125, 255), -1)
        cv2.putText(image, 'REPS', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, 'STAGE', (65, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)

        counterBox.configure(text=str(counter))

        lmain.after(10, update_frame, cap, pose, lmain, counterBox, PRBox, knee_foot_hip_bar, knee_hip_shoulder_bar,table)

    def main():
        """Main function to run the pose detection and UI update loop."""
        window, lmain, counterBox, PRBox, knee_foot_hip_bar, knee_hip_shoulder_bar,table = setup_ui()

        cap = cv2.VideoCapture(camera_index)
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            update_frame(cap, pose, lmain, counterBox, PRBox, knee_foot_hip_bar, knee_hip_shoulder_bar,table)
            window.mainloop()

        cap.release()
        cv2.destroyAllWindows()

    if __name__ == "__main__":
        main()

def code_dips():
    # Initialize MediaPipe pose and drawing utilities
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Initialize global variables
    global counter, stage, PR_rep, series_data, current_series, NH
    counter = 0 
    stage = None
    PR_rep = 0
    series_data = []  # List to store series and their repetitions
    current_series = 1
    NH = "Waiting"

    def setup_ui():
        """Setup the main application window and UI elements."""
        global table
        window = tk.Toplevel()
        window.geometry("1280x720")
        window.title("Dips counter with series table")
        ck.set_appearance_mode("dark")

        # UI Elements
        classLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10, text='Statistics : ')
        classLabel.place(x=10, y=41)

        counterLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10, text='REPS')
        counterLabel.place(x=160, y=1)

        PRLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10, text='PR dips')
        PRLabel.place(x=160, y=580)

        NHlabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 10), text_color="black", padx=10, text='Raise your arms to start (no hands command comming soon)')
        NHlabel.place(x=400, y=1)

        NHbox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="#ff7b00", text=NH)
        NHbox.place(x=500, y=41)

        counterBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="#ff7b00", text='0')
        counterBox.place(x=160, y=41)

        PRBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="Green", text='0')
        PRBox.place(x=160, y=620)

        resetButton = ck.CTkButton(window, text='RESET', command=reset_counter, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="red")
        resetButton.place(x=10, y=620)

        prResetButton = ck.CTkButton(window, text='RESET PR', command=lambda: reset_pr_time(PRBox), height=40, width=120, font=("Arial", 20), text_color="white", fg_color="red")
        prResetButton.place(x=300, y=620)

        # Button to store reps in the series table
        setButton = ck.CTkButton(window, text='SET', command=lambda: store_series(counterBox), height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
        setButton.place(x=450, y=620)

        # Table for series on the right
        tableFrame = tk.Frame(window, bg="black", width=300, height=600)
        tableFrame.place(x=900, y=10)
        tableLabel = ck.CTkLabel(tableFrame, text="Séries et Répétitions", font=("Arial", 16), text_color="white")
        tableLabel.pack()

        table = tk.Text(tableFrame, width=30, height=35, bg="white", fg="black", font=("Arial", 12))
        table.pack()

        frame = tk.Frame(window,height=480, width=480)
        frame.place(x=10, y=90)
        lmain = tk.Label(frame)
        lmain.place(x=0, y=0)

        # Bind the double-click event to the table
        table.bind("<Double-1>", on_double_click)

        return window, lmain, counterBox, PRBox, table, NHbox

    def reset_counter(): 
        """Reset the counter and timer to zero."""
        global counter
        counter = 0 


    def reset_pr_time(PRBox):
        """Reset the PR rep and update the display."""
        global PR_rep
        PR_rep = 0
        PRBox.configure(text='0')

    def calculate_angle(a, b, c):
        """Calculate the angle between three points."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return round(angle, 2)

    def update_table():
        """Update the table with the series data and store it in an Excel file."""
        global series_data, table
        table.delete('1.0', tk.END)  # Clear the table
        
        # Create a DataFrame from the series_data
        df = pd.DataFrame(series_data, columns=['Series', 'Repetitions'])
        
        # Update the table in the GUI
        for series, reps in series_data:
            table.insert(tk.END, f"Série {series}: {reps} répétitions\n")

    def on_double_click(event):
        """Handle double-click event to modify reps."""
        global series_data, table
        # Get the line number where the double-click occurred
        line_index = table.index("@%s,%s" % (event.x, event.y)).split('.')[0]
        line_index = int(line_index) - 1  # Convert to zero-based index
        
        if 0 <= line_index < len(series_data):
            # Prompt the user to enter a new value for reps
            new_reps = simpledialog.askinteger("Input", f"Enter new repetitions for Série {series_data[line_index][0]}:", minvalue=0)
            if new_reps is not None:
                # Update the series_data with the new value
                series_data[line_index] = (series_data[line_index][0], new_reps)
                # Update the table
                update_table()

    def store_series(counterBox):
        """Store the current counter value as a series and reset the counter."""
        global series_data, counter, current_series
        reps = int(counterBox.cget("text"))
        series_data.append((current_series, reps))
        update_table()
        current_series += 1
        reset_counter()
        counterBox.configure(text='0')

    def update_frame(cap, pose, lmain, counterBox, PRBox,table, NHbox):
        """Update the frame for the video capture and UI."""
        ret, frame = cap.read()
        
        if not ret:
            lmain.after(10, update_frame, cap, pose, lmain, counterBox, PRBox,table ,NHbox)
            return
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        try:
            landmarks = results.pose_landmarks.landmark
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            right_foot = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            left_foot = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            left_eye = [landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].y]
            right_eye = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            
            angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            global stage, counter,PR_rep, NH
            if angle > 150 and (left_wrist[1] < left_eye[1] and right_wrist[1] < right_eye[1]): #and NH == "Waiting": # RAJOUT
                #NH = "start" # Faire un visuel pour quand on commence
                print(NH)
            if ankle[1] > hip[1] and hip[1] > left_shoulder[1] and (0 < left_foot[0] < 1) and (0 < right_foot[1] < 1): #and NH == "start":
                if angle > 160 or right_angle > 160:  # Changement de condition ici avec feet_detected
                    stage = "down"
                if stage == 'down' and (angle < 90 or right_angle <90) and (right_shoulder[1] >= right_elbow[1] or left_shoulder[1] >= left_elbow[1]) :
                    stage = "up"
                    counter += 1
                    print(counter)
            if counter > PR_rep:
                PR_rep = counter
                PRBox.configure(text=str(PR_rep))
            if counter != 0  and angle > 160 and (left_wrist[1] < left_shoulder[1] < left_eye[1] or right_wrist[1] < right_shoulder[1] < right_eye[1]) : #RAJOUT and NH == "start"
                #NH = "Waiting"
                store_series(counterBox) # Pas sur ici 


            # Draw amplitude gauge
            min_angle, max_angle = 90, 160
            gauge_value = (angle - min_angle) / (max_angle - min_angle)
            gauge_value = np.clip(gauge_value, 0, 1)
            gauge_height = int(200 * gauge_value)

            cv2.rectangle(image, (20, 300), (60, 300 - gauge_height), (0, 255, 0), -1)
            cv2.rectangle(image, (20, 100), (60, 300), (255, 255, 255), 2)
            cv2.putText(image, 'Amplitude', (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
        except:
            pass

        # Display reps and stage
        cv2.rectangle(image, (0, 0), (225, 73), (0, 125, 255), -1)
        cv2.putText(image, 'REPS', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, 'STAGE', (65, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))               

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        
        counterBox.configure(text=str(counter))
        
        lmain.after(10, update_frame, cap, pose, lmain, counterBox, PRBox,table, NHbox)

    def main():
        """Main function to run the pose detection and UI update loop."""
        window, lmain, counterBox, PRBox, table, NHbox = setup_ui()

        cap = cv2.VideoCapture(camera_index)
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            update_frame(cap, pose, lmain, counterBox, PRBox,table, NHbox)
            window.mainloop()

        cap.release()
        cv2.destroyAllWindows()

    if __name__ == "__main__":
        main()

def open_instagram():
    webbrowser.open("https://www.instagram.com/mathis66mw/")

def update_camera_index(value):
    global camera_index
    camera_index = int(value)

def setup_ui2(root):
    """Configurer l'interface utilisateur de l'application."""
    # Fermer la fenêtre actuelle
    root.destroy()

    # Initialiser camera_index
    global camera_index
    camera_index = 0

    window2 = tk.Tk()
    window2.geometry("1500x900")
    window2.title("Selection window")

    # Créer une liste de boutons ou un menu pour sélectionner un code
    label = ck.CTkLabel(window2, text="Select real-time counter exercise :", font=("Arial", 20),text_color="black")
    label.place(x=160, y=20)

    label2 = ck.CTkLabel(window2, text="Upload video exercise/Analysis :", font=("Arial", 20),text_color="black")
    label2.place(x=600, y=20)

    # Créer un label avec un hyperlien
    labelx = ck.CTkLabel(window2, text="CIApp creator : @mathis66mw", font=("Arial", 10), text_color="blue", cursor="hand2")
    labelx.place(x=150, y=850)
    labelx.bind("<Button-1>", lambda e: open_instagram())

    # Créer des boutons pour chaque code
    btn_analyse = ck.CTkButton(window2, text="Ultimate movement analysis", command=code_analyse_posturale, height=40, width=200, fg_color="red")
    btn_analyse.place(x=500, y=60)

    btn_upload_tractions = ck.CTkButton(window2, text="Upload pull ups video", command=code_upload_tractions, height=40, width=200)
    btn_upload_tractions.place(x=500, y=120)

    btn_upload_squat = ck.CTkButton(window2, text="Upload squats video", command=code_upload_squat, height=40, width=200)
    btn_upload_squat.place(x=500, y=180)

    btn_upload_fl = ck.CTkButton(window2, text="Upload front-lever video", command=code_upload_fl, height=40, width=200)
    btn_upload_fl.place(x=500, y=240)

    btn_uppload_planche = ck.CTkButton(window2, text="Upload planche video", command=code_upload_planche, height=40, width=200)
    btn_uppload_planche.place(x=500, y=300)

    btn_upload_dl = ck.CTkButton(window2, text="More exercises comming soon", command=None, height=40, width=200, fg_color="gray")
    btn_upload_dl.place(x=500, y=360)

    btn_traction = ck.CTkButton(window2, text="Pull ups counter", command=code_traction, height=40, width=200)
    btn_traction.place(x=200, y=60)

    btn_pompes = ck.CTkButton(window2, text="Push ups counter", command=code_pompes, height=40, width=200)
    btn_pompes.place(x=200, y=120)

    btn_squats = ck.CTkButton(window2, text="Squats counter", command=code_squats, height=40, width=200)
    btn_squats.place(x=200, y=180)

    btn_squats = ck.CTkButton(window2, text="Dips counter", command=code_dips, height=40, width=200)
    btn_squats.place(x=200, y=240)

    btn_pose = ck.CTkButton(window2, text="HSPU counter and HS timer", command=code_HSPU, height=40, width=200)
    btn_pose.place(x=200, y=300)

    btn_pose = ck.CTkButton(window2, text="Front-lever timer", command=code_frontlever, height=40, width=200) # Changer le code pour le front-lever
    btn_pose.place(x=200, y=360)

    btn_pose = ck.CTkButton(window2, text="Plank timer", command=code_gainage, height=40, width=200)
    btn_pose.place(x=200, y=420)

    btn_pose = ck.CTkButton(window2, text="L-sit timer", command=code_lsit, height=40, width=200)
    btn_pose.place(x=200, y=480)

    btn_pose = ck.CTkButton(window2, text="Planche timer", command=code_planche, height=40, width=200)
    btn_pose.place(x=200, y=540)

    btn_pose = ck.CTkButton(window2, text="Bench counter", command=code_DC, height=40, width=200, text_color="white", fg_color="green")
    btn_pose.place(x=200, y=600)

    btn_pose = ck.CTkButton(window2, text="Dead-lifts counter", command=code_DL, height=40, width=200, text_color="white", fg_color="green")
    btn_pose.place(x=200, y=660)

    btn_pose = ck.CTkButton(window2, text="Biceps curl counter", command=code_curl, height=40, width=200, text_color="white", fg_color="green")
    btn_pose.place(x=200, y=720)

    btn_enhanced_hs = ck.CTkButton(window2, text="Enhanced handstand timer", command=code_enhanced_HS, height=40, width=200, text_color="white", fg_color="gold")
    btn_enhanced_hs.place(x=200, y=780)

    btn_signaler = ck.CTkButton(window2, text="Report problem", command=open_link, height=40, width=200, text_color="white", fg_color="orange")
    btn_signaler.place(x=1200, y=780)

    camera_slider= tk.Scale(window2, from_=0, to=5, orient="horizontal", length=300, label="Choose camera (0 = default camera)", font=("Arial", 10), command=update_camera_index)
    camera_slider.place(x=1000, y=20)

    label_camera2 = ck.CTkLabel(window2, text="You can download ivcam on your phone and your computer to use it as a camera", font=("Arial", 10), text_color="black")
    label_camera2.place(x=1000, y=90)

    btn_dl_ivcam = ck.CTkButton(window2, text="Download ivcam PC", command=open_link4, height=40, width=200, text_color="white", fg_color="blue")
    btn_dl_ivcam.place(x=1000, y=120)

    btn_openyt = ck.CTkButton(window2, text="Open Youtube tutorial", command=open_link3, height=40, width=200, text_color="white", fg_color="purple")
    btn_openyt.place(x=1200, y=720)


# Interface utilisateur avec Tkinter pour l'authentification
def main():
    root = tk.Tk()
    root.title("Connect to CIApp")

    # Champs de saisie email et mot de passe
    tk.Label(root, text="Email:").grid(row=0, column=0, pady=5, padx=5)
    email_entry = tk.Entry(root, width=30)
    email_entry.grid(row=0, column=1, pady=5, padx=5)

    tk.Label(root, text="Password:").grid(row=1, column=0, pady=5, padx=5)
    password_entry = tk.Entry(root, show="*", width=30)
    password_entry.grid(row=1, column=1, pady=5, padx=5)

    # Boutons pour s'inscrire ou se connecter
    sign_up_btn = tk.Button(root, text="Create account", command=lambda: sign_up(email_entry.get(), password_entry.get()))
    sign_up_btn.grid(row=2, column=0, pady=10)

    log_in_btn = tk.Button(root, text="Login", command=lambda: log_in(email_entry.get(), password_entry.get(), root))
    log_in_btn.grid(row=2, column=1, pady=10)

    root.mainloop()
    

if __name__ == "__main__":
    main()
