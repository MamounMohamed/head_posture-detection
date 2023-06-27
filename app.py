import cv2 as cv , mediapipe as mp , math , os
from flask import Flask, request, render_template , jsonify
app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

@app.route('/')

def home():
       return render_template("video_uploader.html")              

@app.route('/upload_video', methods=['POST'])

def upload_video():
    # Get the uploaded video file
    
    video = request.files['video']
    video_path = os.path.join('uploads', video.filename)
    video.save(video_path)

    
    # Load the video and run emotion detection on each frame
    cap = cv.VideoCapture(video_path)
    
    avg_yaw = 0 
    avg_pitch = 0 
    avg_roll = 0
    prev_yaw = 0 
    prev_pitch = 0 
    prev_roll = 0
    frames = 0 
    
    while cap.isOpened():
        success, image = cap.read()
        
        if not success:
            break

        width = 912
        height = 512
        dim = (width, height)
        
        image = cv.resize(image, dim, interpolation = cv.INTER_AREA)

        # Convert image to RGB
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # Process image
        results = pose.process(image)

        # Draw pose landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if results.pose_landmarks is None:
            continue
        
        landmarks = results.pose_landmarks.landmark
        x = landmarks[mp_pose.PoseLandmark.NOSE].x
        y = landmarks[mp_pose.PoseLandmark.NOSE].y
        z = landmarks[mp_pose.PoseLandmark.NOSE].z
            
            
        pitch = math.degrees(math.atan2(-y, -z))
        yaw = math.degrees(math.atan2(-x, -z))
        roll = math.degrees(math.atan2(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y - landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x - landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x))
            
        print("Yaw:", yaw)
        print("Pitch:", pitch)
        print("Roll:", roll)
        
        avg_yaw += abs(yaw-prev_yaw)
        avg_roll += abs(roll-prev_roll)
        avg_pitch +=abs(pitch-prev_pitch)
            
        prev_roll = roll 
        prev_yaw = yaw 
        prev_pitch = pitch
        frames+=1
        
    cap.release()
    if frames == 0:
        return "Sorry we couldn't capture your head pose"
    
    return jsonify({"Average Turining" : avg_yaw/frames , 
              "Average Nodding" : avg_pitch/frames , 
              "Average Tilting" : avg_roll/frames}
    )
            
if __name__ == '__main__':
    app.run (debug = True)


            
