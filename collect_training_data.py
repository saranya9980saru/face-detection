import cv2

# Method to generate dataset to recognize a person
def generate_dataset(img, id, img_id):
    cv2.imwrite(f"data/user.{str(id)}.{str(img_id)}.jpg", img)

# Method to draw boundary around the detected feature
def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    print(f"Found {len(features)} features.")  # Debugging print
    coords = []
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, text, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords

# Method to detect the features
def detect(img, faceCascade, img_id):
    color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0), "white": (255, 255, 255)}
    coords = draw_boundary(img, faceCascade, 1.1, 10, color['blue'], "Face")
    if len(coords) == 4:
        roi_img = img[coords[1]:coords[1] + coords[3], coords[0]:coords[0] + coords[2]]
        user_id = 1  # Assign unique id to each user
        generate_dataset(roi_img, user_id, img_id)
    return img

# Loading classifiers
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
if faceCascade.empty():
    print("Error loading cascade classifier.")
    exit()

# Capturing real time video stream
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Error: Unable to access camera.")
    exit()

img_id = 0

while True:
    _, img = video_capture.read()
    if img is None:
        print("Failed to capture image.")
        break
    if img_id % 50 == 0:
        print(f"Collected {img_id} images")
    img = detect(img, faceCascade, img_id)
    cv2.imshow("face detection", img)
    img_id += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# releasing web-cam
video_capture.release()
cv2.destroyAllWindows()
