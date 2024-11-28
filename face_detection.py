import cv2

# Method to draw boundary around the detected feature
def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, text, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords

# Method to detect features
def detect(img, faceCascade, eyeCascade, noseCascade, mouthCascade):
    color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0), "white": (255, 255, 255)}
    coords = draw_boundary(img, faceCascade, 1.1, 10, color['blue'], "Face")
    if len(coords) == 4:
        roi_img = img[coords[1]:coords[1] + coords[3], coords[0]:coords[0] + coords[2]]
        draw_boundary(roi_img, eyeCascade, 1.1, 12, color['red'], "Eye")
        draw_boundary(roi_img, noseCascade, 1.1, 4, color['green'], "Nose")
        draw_boundary(roi_img, mouthCascade, 1.1, 20, color['white'], "Mouth")
    return img

# Loading classifiers
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyesCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
noseCascade = cv2.CascadeClassifier('Nariz.xml')
mouthCascade = cv2.CascadeClassifier('Mouth.xml')

# Validate classifiers
if faceCascade.empty():
    print("Error loading haarcascade_frontalface_default.xml")
if eyesCascade.empty():
    print("Error loading haarcascade_eye.xml")
if noseCascade.empty():
    print("Error loading Nariz.xml")
if mouthCascade.empty():
    print("Error loading Mouth.xml")

# Capturing real-time video stream
video_capture = cv2.VideoCapture(0)  # Use 0 for built-in webcam

while True:
    ret, img = video_capture.read()
    if not ret or img is None:
        print("Failed to capture frame from camera. Exiting...")
        break
    img = detect(img, faceCascade, eyesCascade, noseCascade, mouthCascade)
    cv2.imshow("face detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
