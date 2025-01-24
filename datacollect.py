import cv2
import os

# Initialize video capture
video = cv2.VideoCapture(0)

# Load Haar cascade for face detection
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to create a new folder for a new person
def create_folder(name):
    path = f"datasets/{name}"
    if not os.path.exists(path):
        os.makedirs(path)
    return path

# Function to capture images for a new person
def capture_images(name):
    folder_path = create_folder(name)
    count = 0
    while True:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = facedetect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            count += 1
            img_path = f"{folder_path}/{name}.{count}.jpg"
            cv2.imwrite(img_path, gray[y:y+h, x:x+w])
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Capturing {name}'s face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Frame", frame)
        
        # Stop capturing after 200 images
        if count >= 200:
            print("Data Collection Completed")
            break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

name = input("Enter the name of the person: ")
capture_images(name)

video.release()
cv2.destroyAllWindows()