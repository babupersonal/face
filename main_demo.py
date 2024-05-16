from flask import Flask, render_template, Response, request
from flask import redirect, url_for
import os
import cv2
import dlib
import numpy as np
import time

app = Flask(__name__)

predictor_path = "face_function/shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "face_function/dlib_face_recognition_resnet_model_v1.dat"
faces_folder_path = "faceDB"

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

descriptors = []
FaceName = []
difference = 0.5

# Load and process each face image from the database
for file in os.listdir(faces_folder_path):
    if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
        file_path = os.path.join(faces_folder_path, file)
        FaceName.append(os.path.splitext(file)[0])
        img = cv2.imread(file_path)
        if img is not None:
            try:
                dets = detector(img, 0)
                for k, d in enumerate(dets):
                    shape = sp(img, d)
                    face_descriptor = facerec.compute_face_descriptor(img, shape)
                    descriptors.append(np.array(face_descriptor))
            except Exception as e:
                print(f"偵測特徵和提取特徵失敗: {str(e)}")
        else:
            print(f"無法讀取圖片檔案: {file_path}")

def gen_frames():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        img = cv2.flip(img, 1)
        stime = time.time()
        dets = detector(img, 0)

        for k, d in enumerate(dets):
            shape = sp(img, d)
            face_descriptor = facerec.compute_face_descriptor(img, shape)
            d_test = np.array(face_descriptor)
            x1, y1, x2, y2 = d.left(), d.top(), d.right(), d.bottom()

            # Calculate distance and identify if known or unknown
            dist = [np.linalg.norm(i - d_test) for i in descriptors]
            candidate_dist = dict(zip(FaceName, dist))
            rec_name = min(candidate_dist, key=candidate_dist.get)

            box_color = (0, 255, 0)  # Default to green
            status_text = "Pass"

            if candidate_dist[rec_name] > difference:
                rec_name = "unknown"
                status_text = "Fail"
                box_color = (0, 0, 255)  # Change to red for Fail

            cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 4)
            cv2.putText(img, status_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 3)
            cv2.putText(img, rec_name, (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        etime = time.time()
        fps = round(1 / (etime - stime), 2)
        cv2.putText(img, f"FPS: {fps}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/index')
def home():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/take_photo', methods=['POST'])
def take_photo():
    cap = cv2.VideoCapture(0)
    ret, img = cap.read()
    cap.release()

    # Save the image to faceDB directory
    if ret:
        count = len(os.listdir(faces_folder_path)) + 1
        img_path = os.path.join(faces_folder_path, f"{count}.jpg")
        cv2.imwrite(img_path, img)
        return "Photo Taken Successfully!"
    else:
        return "Failed to Take Photo"

if __name__ == '__main__':
    app.run(debug=True, port=8000)

