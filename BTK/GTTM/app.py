from flask import Flask, render_template, Response, jsonify
import cv2
import threading
import time

from traffic_violation_detector import ViolationDetector

app = Flask(__name__)
detector = ViolationDetector(video_source='C:/Users/hi/Downloads/Smart-City-and-Smart-Agriculture-main/Smart_City-Case_study/hi2.mp4')

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    while True:
        frame = detector.get_next_frame()
        if frame is None:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ✅ API để frontend lấy danh sách vi phạm
@app.route('/violations')
def violations():
    return jsonify(detector.get_violations())

if __name__ == '__main__':
    app.run(debug=True)
