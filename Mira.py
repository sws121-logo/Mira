import speech_recognition as sr
from sentence_transformers import SentenceTransformer, util
from langchain.llms import HuggingFaceHub
from gtts import gTTS
import pygame
from io import BytesIO
import tkinter as tk
from tkinter import Text, Scrollbar, Button
from threading import Thread
import os
import cv2
import threading
import mediapipe as mp
import time
import math
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageTk
import webbrowser


model = YOLO('yolov8s.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
              "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
              "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
              "hot dog", "pizza", "donut", "cake", "chain", "sofa", "pottedplant", "bed", "diningtable", "toilet",
              "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
              "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

# variables
frame_counter = 0

# constants
FONTS = cv2.FONT_HERSHEY_COMPLEX

# face bounder indices
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176,
             149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# Lips indices for Landmarks
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40,
        39,
        37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]

LOWER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS = [185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]

# Left eyes indices
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]

# Right eyes indices
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]

mouth_open_threshold = 1
mouth_closed_time_threshold = 10


class VirtualInterviewerApp:

    def __init__(self, root,screen_width,screen_height):

        self.master = root

        self.fr = tk.Frame(master,bg="#abcdef")
        self.fr.grid(row=1, column=1, columnspan=4)

        self.fr1=tk.Frame(self.master,bg="#abcdef")
        self.fr1.grid(row=0,column=0,rowspan=2)

        self.logo_fr = tk.Frame(self.master, bg="#abcdef")
        self.logo_fr.grid(row=2, column=1,columnspan=4,rowspan=4)

        self.link_label = tk.Label(self.fr1,text="Mira",font=('cursive',28,'italic'),fg="blue",bg="#abcdef")
        self.link_label.grid(pady=10, row=0, column=0)
        self.link_label.bind("<Button-1>", lambda event: self.open_link())

        self.label_widget = tk.Label(self.fr1,justify="left", text="Welcome to our Desktop Mock Interview Application\nyour gateway"
                                                " to honing your interview skills "
                                                "and boosting\nyour confidence!Whether you're preparing for your dream\njob or just aiming to enhance"
                                                "your overall interview\nperformance, you're in the right place."
                                                "\n\nOur user-friendly interface and realistic interview simulations "
                                                "\nprovide a dynamic platform to practice and refine your responses. "
                                                "\nTailor your experience by selecting from a range of industries an\njob positions,"
                                                "and receive instant feedback to help you identify\nareas for improvement."
                                                "\n\nEmbark on a journey of self-discovery and professional\ngrowth as you navigate "
                                                "through challenging questions, "
                                                "\nreplicate real interview scenarios, and build the resilience\nneeded to "
                                                "ace any job interview. Let's turn\nthose nerves into strengths and transform your "
                                                "\ninterview apprehension into interview excellence."
                                                "\n\nGet ready to shine in your next interview – your success \nstory starts here!",bg="#abcdef",
                                     font=('Arail',10))
        self.label_widget.grid(padx=20, row=2, column=0,pady=5)

        self.text_widget = Text(self.master, wrap="word", width=98, height=20,font=(10),bg="white")
        self.text_widget.grid(pady=10,row=0,column=1)
        self.text_widget.config()

        self.start_button = Button(self.fr, text="Start Interview", bg="#93fc8b", font=('Arail',12),command=self.function_call)
        self.start_button.grid(row=0, column=0, padx=10)

        self.exit_btn = tk.Button(self.fr, text="Exit", bg="#93fc8b", font=('Arail',12), command=self.master.quit,width=12)
        self.exit_btn.grid(row=0, column=1, padx=10)

        os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
        self.questions = []
        self.flag = 0
        self.stop_pressed = False

    def open_link(self):
        webbrowser.open("https://www.theiotacademy.co/")

    def speech_to_text(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            self.text_widget.insert(tk.END, "Start Speaking\n")

            try:
                audio = recognizer.listen(source)
                text = recognizer.recognize_google(audio)
                return text
            except sr.UnknownValueError:
                return 'No Audio'
            except sr.RequestError as e:
                return 'No Audio'

    def text_to_audio(self, text):
        language = 'en'
        tts = gTTS(text=text, lang=language, slow=False)
        audio_stream = BytesIO()
        tts.write_to_fp(audio_stream)

        pygame.mixer.init()
        audio_stream.seek(0)
        pygame.mixer.music.load(audio_stream)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

    def interview_question(self, question):

        result = self.llm(question)[1:]
        self.text_widget.insert(tk.END, f"Question: {question}\n\n")
        self.text_to_audio(question)
        user_input = self.speech_to_text()
        self.text_widget.insert(tk.END, f"User Answer: {user_input}\n\n\n")
        self.compare_answers(result, user_input)

    def compare_answers(self, result, user_input):
        embedding1 = self.model.encode(result, convert_to_tensor=True)
        embedding2 = self.model.encode(user_input, convert_to_tensor=True)
        cosine_similarity = util.pytorch_cos_sim(embedding1, embedding2)
        if cosine_similarity.item() > 0.5:
            self.flag += 1

    def interview_process(self):
        self.flag = 0
        for question in self.questions:
            self.interview_question(question)

            # Check if the stop button has been pressed
            if self.stop_pressed:
                break

        self.text_widget.insert(tk.END, f"Total Correct Answers: {self.flag}\n\n")
        print(self.flag)

        # Enable the start button and disable the stop button after the interview ends
        self.start_button.config(state=tk.NORMAL)
        # self.stop_button.config(state=tk.DISABLED)

    def function_call(self):
        thread = threading.Thread(target=self.worker, args=(self.master, self.fr, self.fr1,))
        thread.start()
        self.start_interview()

    def start_interview(self):

        self.text_widget.delete(1.0, tk.END)  # Clear previous content
        self.text_widget.insert(tk.END, "Interview Started...\n\n")
        INTRODUCTION_TEXT = "Hello, I am your virtual interviewer Mira. I am here to help you enhance your data science skills. " \
                        "You can start speaking after I complete the question. Best of Luck!"
        self.text_to_audio(INTRODUCTION_TEXT)

        self.model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        self.llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature": 0.6, "max_length": 100000, "max_new_tokens": 1000})

        # Generate questions
        query_result = self.llm('generate 10 very basic data science interview question, give only questions, questions are generated randomly, question should be diverse and based on basis')

        # Split the result into a list of questions
        self.questions = query_result.split('\n')[1:]
        self.questions = [question.split('. ', 1)[1] for question in self.questions]

        # Run the interview process in a separate thread
        interview_thread = Thread(target=self.interview_process)
        interview_thread.start()

    def rescale(self,img, scale=0.6):
        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)
        dimension = (width, height)
        return cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)

    def detecting_mobile_and_people_count(self, img, start_time1=0, start_time2=0, time1=0, time2=0):
        results = model(img, stream=True)
        people_count = 0
        mobile_detected = False

        for r in results:
            boxes = r.boxes
            # iterating through every detection one by one
            for box in boxes:

                cls = int(box.cls[0])  # changing the class number from tensor to integer
                label = classNames[cls]  # retrieving the class name
                conf_score = int(box.conf[0] * 100)

                # Checking the labels if a person or cell phone has been detected,
                # if a person is detected then counting the number of people

                if label == 'person' and conf_score > 60:
                    people_count += 1
                elif label == 'cell phone':
                    mobile_detected = True
                    if not start_time1:
                        start_time1 = time.time()

            # checking if there is more than one person in the fame, then show the error
            if people_count > 1:
                cv2.putText(img, "Warning: More than one person", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 1)

            if people_count == 0:
                cv2.putText(img, f"Face not detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

        if mobile_detected:

            time1 = time.time() - start_time1
            # cv2.putText(img, f" Warning: Mobile Phone detected", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif mobile_detected == False:
            start_time1 = 0
            time1 = 0

        return img, start_time1, time1

    # Landmark detection function
    def landmarksDetection(self, img, results, draw=False):
        img_height, img_width = img.shape[:2]
        # List of (x, y) coordinates
        mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in
                      results.multi_face_landmarks[0].landmark]
        if draw:
            [cv2.circle(img, p, 2, (0, 255, 0), -1) for p in mesh_coord]

        # Return the list of tuples for each landmark
        return mesh_coord

    # Euclidean distance
    def euclideanDistance(self, point, point1):
        x, y = point
        x1, y1 = point1
        distance = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
        return distance

    # Eyes Extractor function
    def eyesExtractor(self, img, right_eye_coords, left_eye_coords):
        # Convert color image to grayscale image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Get the dimensions of the image
        dim = gray.shape

        # Create a mask from grayscale dimensions
        mask = np.zeros(dim, dtype=np.uint8)

        # Draw eye shape on mask with white color
        cv2.fillPoly(mask, [np.array(right_eye_coords, dtype=np.int32)], 255)
        cv2.fillPoly(mask, [np.array(left_eye_coords, dtype=np.int32)], 255)

        # Draw eyes image on the mask, where the white shape is
        eyes = cv2.bitwise_and(gray, gray, mask=mask)
        eyes[mask == 0] = 155

        # Get minimum and maximum x and y for right and left eyes
        # For Right Eye
        r_max_x = (max(right_eye_coords, key=lambda item: item[0]))[0]
        r_min_x = (min(right_eye_coords, key=lambda item: item[0]))[0]
        r_max_y = (max(right_eye_coords, key=lambda item: item[1]))[1]
        r_min_y = (min(right_eye_coords, key=lambda item: item[1]))[1]

        # For Left Eye
        l_max_x = (max(left_eye_coords, key=lambda item: item[0]))[0]
        l_min_x = (min(left_eye_coords, key=lambda item: item[0]))[0]
        l_max_y = (max(left_eye_coords, key=lambda item: item[1]))[1]
        l_min_y = (min(left_eye_coords, key=lambda item: item[1]))[1]

        # Crop the eyes from the mask
        cropped_right = eyes[r_min_y: r_max_y, r_min_x: r_max_x]
        cropped_left = eyes[l_min_y: l_max_y, l_min_x: l_max_x]

        # Return the cropped eyes
        return cropped_right, cropped_left

    # Eyes Position Estimator
    def positionEstimator(self, cropped_eye):
        # Get height and width of the eye
        h, w = cropped_eye.shape

        # Remove noise from images
        gaussian_blur = cv2.GaussianBlur(cropped_eye, (9, 9), 0)
        median_blur = cv2.medianBlur(gaussian_blur, 3)

        # Apply thresholding to convert to a binary image
        ret, threshed_eye = cv2.threshold(median_blur, 130, 255, cv2.THRESH_BINARY)

        # Create fixed parts for the eye
        piece = int(w / 3)

        # Slice the eye into three parts
        right_piece = threshed_eye[0:h, 0:piece]
        center_piece = threshed_eye[0:h, piece: piece + piece]
        left_piece = threshed_eye[0:h, piece + piece:w]

        # Call pixel counter function
        eye_position, color = self.pixelCounter(right_piece, center_piece, left_piece)

        return eye_position, color

    # Pixel counter function
    def pixelCounter(self, first_piece, second_piece, third_piece):
        # Count black pixels in each part
        right_part = np.sum(first_piece == 0)
        center_part = np.sum(second_piece == 0)
        left_part = np.sum(third_piece == 0)
        # Create a list of these values
        eye_parts = [right_part, center_part, left_part]

        # Get the index of the max value in the list
        max_index = eye_parts.index(max(eye_parts))
        pos_eye = ''
        color = (255, 0, 0)
        if max_index == 0:
            pos_eye = "RIGHT"
            # color = [utils.BLACK, utils.GREEN]
        elif max_index == 1:
            pos_eye = 'CENTER'
            # color = [utils.YELLOW, utils.PINK]
        elif max_index == 2:
            pos_eye = 'LEFT'
            # color = [utils.GRAY, utils.YELLOW]
        else:
            pos_eye = "Closed"
            # color = [utils.GRAY, utils.YELLOW]
        return pos_eye, color

    def initialize_camera(self,video_source=0):
        vid = cv2.VideoCapture(video_source)
        return vid

    map_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils

    def update(self,canvas, vid):
        flag = 0
        mouth_closed_start_time = None
        start_time1, time1 = 0, 0
        COUNT = 0
        with self.map_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
            ret, frame = vid.read()
            if ret:
                frame = self.rescale(frame)
                img_after_detection, start_time1, time1 = self.detecting_mobile_and_people_count(frame, start_time1, time1)
                time1 = int(time1)

                image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                image.flags.writeable = False

                results = face_mesh.process(frame_rgb)
                result = face_mesh.process(image)

                image.flags.writeable = True

                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                img_h, img_w, img_c = image.shape
                face_3d = []
                face_2d = []

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        for idx, lm in enumerate(face_landmarks.landmark):
                            if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                                if idx == 1:
                                    nose_2d = (lm.x * img_w, lm.y * img_h)
                                    nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                                x, y = int(lm.x * img_w), int(lm.y * img_h)

                                # Get the 2D Coordinates
                                face_2d.append([x, y])

                                # Get the 3D Coordinates
                                face_3d.append([x, y, lm.z])

                                # Convert it to the NumPy array
                        face_2d = np.array(face_2d, dtype=np.float64)

                        # Convert it to the NumPy array
                        face_3d = np.array(face_3d, dtype=np.float64)

                        # The camera matrix
                        focal_length = 1 * img_w

                        cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                               [0, focal_length, img_w / 2],
                                               [0, 0, 1]])

                        # The Distance Matrix
                        dist_matrix = np.zeros((4, 1), dtype=np.float64)

                        # Solve PnP
                        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                        # Get rotational matrix
                        rmat, jac = cv2.Rodrigues(rot_vec)

                        # Get angles
                        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                        # Get the y rotation degree
                        x = angles[0] * 360
                        y = angles[1] * 360

                        # See where the user's head is tilting
                        if y < -10:
                            head_pose_text = "PLEASE LOOK STRAIGHT"
                            COUNT += 1

                        elif y > 10:
                            head_pose_text = "PLEASE LOOK STRAIGHT"
                            COUNT += 1

                        elif x > 15:
                            head_pose_text = "PLEASE LOOK STRAIGHT"
                            COUNT += 1

                        elif x < -10:
                            head_pose_text = "PLEASE LOOK STRAIGHT"
                            COUNT += 1

                        else:
                            head_pose_text = "  "
                            COUNT = 0

                        nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix,
                                                                         dist_matrix)

                        p1 = (int(nose_2d[0]), int(nose_2d[1]))
                        p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))

                        cv2.line(image, p1, p2, (255, 0, 0), 2)

                        # Add the text on the image
                        # cv2.putText(image, head_pose_text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        # cv2.putText(frame, head_pose_text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                if results.multi_face_landmarks:
                    for landmarks in results.multi_face_landmarks:
                        # Extract landmarks for the upper and lower lips
                        upper_lip_landmarks = landmarks.landmark[0]
                        lower_lip_landmarks = landmarks.landmark[87]

                        # Calculate the distance between the upper and lower lip landmarks
                        lip_distance = abs(upper_lip_landmarks.y - lower_lip_landmarks.y) * 100
                        # print(int(lip_distance))

                        mouth_open = lip_distance > 2

                        if mouth_open:
                            mouth_closed_start_time = None
                        else:
                            if mouth_closed_start_time is None:

                                mouth_closed_start_time = time.time()

                            else:
                                elapsed_time = time.time() - mouth_closed_start_time

                                if elapsed_time > mouth_closed_time_threshold:
                                    flag = 1
                                    break

                        # Display the mouth openness status
                        status_text = " " if mouth_open else "PLEASE SPEAK"

                    # cv2.putText(frame, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(img_after_detection, cv2.COLOR_BGR2RGB)))
                canvas.create_image(0, 0, image=photo, anchor=tk.NW)
                canvas.photo = photo  # Store a reference to avoid garbage collection
            canvas.after(10, lambda: self.update(canvas, vid))

    def close_camera(vid):
        if vid.isOpened():
            vid.release()

    def stop_interview(self):
        # Set a flag to indicate that the stop button has been pressed
        self.stop_pressed = True

    def worker(self,window,fr,fr1):
        video_source = 0
        vid = self.initialize_camera(video_source)
        canvas = tk.Canvas(fr1,bg="#abcdef")
        canvas.grid(row=1,column=0,pady=5)

        # exit_btn = tk.Button(fr, text="Exit", bg="#fa968e",font=(12),command=window.quit).grid(row=0,column=1,columnspan=3,padx=10)

        self.update(canvas, vid)
        window.protocol("WM_DELETE_WINDOW", lambda: self.close_camera(vid))

class about_page:
    def __int__(self,about_frame):
        self.frame=about_frame
        lb = tk.Label(self.frame, text="This page is about", font=('Bold', 30))
        lb.pack()
        self.frame.pack()

if __name__ == "__main__":
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.geometry(f"{screen_width}x{screen_height}")
    root.configure(bg="#abcdef")
    root.title("Mira")

    nav_frame = tk.Frame(root, bg="#ebe9e6")
    nav_frame.pack(side=tk.TOP)
    nav_frame.pack_propagate(False)
    nav_frame.configure(width=screen_width, height=40)

    master = tk.Frame(root, bg="#abcdef")
    master.pack()
    master.pack_propagate(False)
    master.configure(width=screen_width, height=screen_height - 40)

    # img = ImageTk.PhotoImage(Image.open("wallpaper.jpeg"))
    # label = tk.Label(master, image=img)
    # label.pack()
    label = tk.Label(master, text="Welcome to your online Interviewer")
    label.pack()

    home_btn = tk.Button(nav_frame, text='Practice', font=('Bold', 12),
                              bd=0, bg='#ebe9e6', command=lambda: indicate(home_indicator, home))
    home_btn.place(x=100, y=5)
    home_indicator = tk.Label(nav_frame, text='', bg='#ebe9e6')
    home_indicator.place(x=100, y=3, width=50, height=3)

    about_btn = tk.Button(nav_frame, text='About', font=('Bold', 12),
                               bd=0, bg='#ebe9e6'
                               , command=lambda: indicate(about_indicator, about))

    about_btn.place(x=200, y=5)
    about_indicator = tk.Label(nav_frame, text='', bg='#ebe9e6')
    about_indicator.place(x=200, y=3, width=50, height=3)

    contact_btn = tk.Button(nav_frame, text='Contact', font=('Bold', 12),
                                 bd=0, bg='#ebe9e6'
                                 , command=lambda: indicate(contact_indicator, contact))
    contact_btn.place(x=300, y=5,)
    contact_indicator = tk.Label(nav_frame, text='', bg='#ebe9e6')
    contact_indicator.place(x=300, y=3, width=70, height=3)


    def indicate(lb,page):
        hide_indicate()
        lb.config(bg='blue')
        delete_pages()
        page()

    def hide_indicate():
        home_indicator.config(bg="#ebe9e6")
        about_indicator.config(bg="#ebe9e6")
        contact_indicator.config(bg="#ebe9e6")

    def delete_pages():
        for frame in master.winfo_children():
            frame.destroy()

    def home():
        VirtualInterviewerApp(master,screen_width,screen_height)

    def about():
        about_frame=tk.Frame(master)
        # about_page(about_frame)

        webbrowser.open("https://www.theiotacademy.co/")
        lb = tk.Label(about_frame, text="You are redirected to IOT Academy Home Page", font=('Bold', 20),bg="#abcdef")
        lb.pack()
        about_frame.pack()

    def contact():
        contact_frame=tk.Frame(master)
        webbrowser.open("https://www.theiotacademy.co/contact")
        lb=tk.Label(contact_frame,text="You have been redirected to the cotact page of IOT Academy",font=('Bold',20),bg="#abcdef")
        lb.pack()
        contact_frame.pack()




    root.mainloop()