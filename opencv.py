import cv2
from model.temp.facereg import recognize_face
from model.temp.emotion import detect_emotion

def process_video(video_path):
    # Load the pre-trained face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file is opened successfully
    if not cap.isOpened():
        print("Error: Couldn't open the video file.")
        return
    
        # Set the window size (you can adjust these values as needed)
    window_width = 1000
    window_height =700

    # Set the window properties to allow resizing
    cv2.namedWindow('Face Recognition', cv2.WINDOW_NORMAL)

    # Resize the window to the desired size
    cv2.resizeWindow('Face Recognition', window_width, window_height)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If the frame is not read properly, break the loop
        if not ret:
            break

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Extract faces, perform face recognition and emotion detection
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            identity = recognize_face(face_img)  # Perform face recognition
            emotion = detect_emotion(face_img)  # Perform emotion detection
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{identity} - {emotion}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame with face detection, recognition, and emotion detection
        cv2.imshow('Face Recognition', frame)

        # Press 'q' to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Release the video capture object
    cap.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Path to the video file
    video_path = 'data/classroom/video2.mp4'

    # Process the video
    process_video(video_path)
