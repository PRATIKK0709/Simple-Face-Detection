import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

image_path = './all.png' 
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

num_faces = len(faces)
text = f'Detected Faces: {num_faces}'
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2
font_color = (0, 255, 0)
cv2.putText(image, text, (10, 30), font, font_scale, font_color, font_thickness, lineType=cv2.LINE_AA)

for i, (x, y, w, h) in enumerate(faces):
    face = image[y:y + h, x:x + w]
    cv2.imwrite(f'face_{i}.jpg', face) 

    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    coordinates_text = f'Face {i+1}: ({x}, {y})'
    cv2.putText(image, coordinates_text, (x, y - 10), font, 0.5, font_color, 1, lineType=cv2.LINE_AA)

cv2.namedWindow('Detected Faces', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Detected Faces', 1080, 700)

cv2.imshow('Detected Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
