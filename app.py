import cv2
from deepface import DeepFace

# Replace the camera stream URL with the default webcam ID
camera_device_id = 0  # 0 for the default webcam, use 1 or other numbers for additional webcams

# Start capturing the video stream from the webcam
cap = cv2.VideoCapture(camera_device_id)

# Reduce buffering in OpenCV to minimize lag (optional for webcam, but keeping it for consistency)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to exit the program.")

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face region of interest (ROI)
        face_roi = frame[y:y+h, x:x+w]

        try:
            # Analyze emotions using DeepFace
            result = DeepFace.analyze(face_roi, actions=["emotion"], enforce_detection=False)

            # Check if the result is a list
            if isinstance(result, list):
                result = result[0]  # Use the first result

            # Get the emotions and their scores from the analysis result
            emotions = result.get("emotion", {})

            # Find the dominant emotion
            max_emotion = max(emotions, key=emotions.get)
            max_score = emotions[max_emotion]

            # Customize the message based on the dominant emotion
            if max_emotion == "happy":
                message = f"Happiness: {max_score:.2f}"
            elif max_emotion == "surprise":
                message = f"Surprise: {max_score:.2f}"
            else:
                # If any emotion other than 'happy' is dominant, display a message encouraging a smile
                message = "Smile a little, it looks good on you!"

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Display the message above the rectangle
            cv2.putText(frame, message, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        except Exception as e:
            print(f"Error analyzing emotion: {e}")

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close OpenCV windows
cap.release()
cv2.destroyAllWindows()