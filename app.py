import os
import cv2
import numpy as np
import pandas as pd
import face_recognition
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from threading import Thread

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for the app

# Specify paths for dataset, attendance records, and courses file
dataset_path = "C:/Users/Dev Patel/OneDrive/Desktop/Dataset"
attendance_dir = "C:/Users/Dev Patel/OneDrive/Desktop/Attendance"
courses_file = "C:/Users/Dev Patel/OneDrive/Desktop/Courses.csv"

# Ensure necessary directories exist
for dir_path in [attendance_dir, os.path.dirname(courses_file)]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Create a blank courses file if it doesn't exist
if not os.path.exists(courses_file):
    pd.DataFrame(columns=['Course Name', 'Course Code']).to_csv(courses_file, index=False)

# Load dataset images and corresponding class names (labels)
images = []
classNames = []
print("Loading dataset images...")
for cl in os.listdir(dataset_path):
    curImg = cv2.imread(f'{dataset_path}/{cl}')  # Read each image
    if curImg is not None:  # Ensure the image is valid
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])  # Extract the file name (without extension) as the label
print(f"Loaded {len(images)} images from the dataset.")

# Function to encode faces from the dataset images
def findEncodings(images):
    print("Encoding dataset images...")
    encodeList = []
    for img in images:
        try:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image to RGB format
            encodings = face_recognition.face_encodings(rgb_img)  # Generate face embeddings
            if encodings:  # Ensure at least one face is detected
                encodeList.append(encodings[0])
        except Exception as e:
            print(f"Error encoding image: {e}")  # Log errors during encoding
    print(f"Encoding complete. Total encodings: {len(encodeList)}")
    return encodeList

# Pre-compute encodings for all known faces
encodeListKnown = findEncodings(images)

# Function to mark attendance for a recognized individual
def markAttendance(name, course):
    # Generate a file path for the attendance record of the current date
    today_date = datetime.now().strftime("%Y-%m-%d")
    attendance_path = os.path.join(attendance_dir, f"Attendance_{course}_{today_date}.csv")

    # Create the attendance file with headers if it doesn't exist
    if not os.path.exists(attendance_path):
        with open(attendance_path, 'w') as f:
            f.write("Name,Course,Time,Date\n")

    # Append attendance details if the person hasn't been marked yet
    with open(attendance_path, 'r+') as f:
        lines = f.readlines()
        nameList = {line.split(',')[0] for line in lines}  # Extract names already marked
        if name not in nameList:
            now = datetime.now()
            f.write(f'{name},{course},{now.strftime("%H:%M:%S")},{now.strftime("%Y-%m-%d")}\n')
            print(f"Marked attendance for {name} in {course}")

# Global flags and variables for video processing
video_running = False
video_capture = None
video_thread = None
current_course = None

# Function to start video capture and processing for attendance
def startVideoCapture(course):
    global video_running, video_capture, video_thread, current_course
    current_course = course  # Set the current course for attendance

    if video_running:
        print("Video feed already running.")
        return

    video_running = True
    video_capture = cv2.VideoCapture(0)  # Open the webcam
    print(f"Video feed started for course: {course}")
    
    video_thread = Thread(target=processVideoFeed)  # Start a separate thread for processing
    video_thread.start()

# Function to stop video capture and clean up resources
def stopVideoCapture():
    global video_running, video_capture, video_thread
    if video_capture:
        video_running = False
        video_capture.release()  # Release the webcam
        video_capture = None
        print("Video feed stopped.")
        if video_thread and video_thread.is_alive():
            video_thread.join()  # Wait for the thread to finish
    cv2.destroyAllWindows()  # Close all OpenCV windows

# Function to process the video feed and recognize faces
def processVideoFeed():
    global video_running, video_capture, current_course

    print(f"Processing video feed for attendance in {current_course}...")
    while video_running:
        if video_capture is None:
            break

        success, img = video_capture.read()  # Capture a frame from the webcam
        if not success:
            print("Failed to read video frame.")
            break

        # Resize and convert the frame to RGB for faster processing
        imgS = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        imgS_rgb = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        # Detect faces and compute encodings for the current frame
        facesCurFrame = face_recognition.face_locations(imgS_rgb)
        encodeCurFrame = face_recognition.face_encodings(imgS_rgb, facesCurFrame)

        for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
            faceDist = face_recognition.face_distance(encodeListKnown, encodeFace)  # Compute distances
            matchIndex = np.argmin(faceDist)  # Find the best match

            # Determine if the face is recognized based on a threshold
            if faceDist[matchIndex] < 0.6:
                accuracy = (1 - faceDist[matchIndex]) * 100
                if accuracy > 50:  # Ensure sufficient accuracy
                    name = classNames[matchIndex].upper()
                else:
                    name = "Unknown"
            else:
                name = "Unknown"

            # Draw rectangles and labels around the face
            y1, x2, y2, x1 = [v * 4 for v in faceLoc]  # Scale up coordinates
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            # Mark attendance for recognized individuals
            if name != "Unknown":
                markAttendance(name, current_course)

        cv2.imshow('Video', img)  # Display the video feed
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Stop on 'q' key press
            break

# Flask route to render the main page
@app.route('/')
def index():
    courses_df = pd.read_csv(courses_file)  # Load courses for dropdown menu
    return render_template('index.html', courses=courses_df.to_dict('records'))

# Route to add a new course
@app.route('/add_course', methods=['POST'])
def add_course():
    try:
        course_name = request.form.get('course_name')
        course_code = request.form.get('course_code')
        
        courses_df = pd.read_csv(courses_file)
        new_course = pd.DataFrame({'Course Name': [course_name], 'Course Code': [course_code]})
        courses_df = pd.concat([courses_df, new_course], ignore_index=True)
        courses_df.to_csv(courses_file, index=False)
        
        return jsonify({"status": "success", "message": "Course added successfully!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# Route to start the attendance process
@app.route('/start', methods=['POST'])
def startAttendance():
    try:
        course = request.form.get('course')
        startVideoCapture(course)
        return jsonify({"status": "success", "message": f"Attendance started for {course}"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# Route to stop the attendance process
@app.route('/stop', methods=['POST'])
def stopAttendance():
    try:
        stopVideoCapture()
        return jsonify({"status": "success", "message": "Attendance process stopped."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# Route to view attendance records
@app.route('/attendance')
def viewAttendance():
    try:
        attendance_files = [f for f in os.listdir(attendance_dir) if f.startswith('Attendance_')]  # List attendance files
        return render_template('attendance.html', files=attendance_files)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# Route to download an attendance file
@app.route('/download_attendance/<filename>')
def download_attendance(filename):
    try:
        filepath = os.path.join(attendance_dir, filename)
        return send_file(filepath, as_attachment=True)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# Start the Flask app
if __name__ == "__main__":
    app.run(debug=True)
