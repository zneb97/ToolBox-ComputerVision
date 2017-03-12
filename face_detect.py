import numpy as np
import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('/home/zneb/SoftDes/haarcascade_frontalface_alt.xml')
kernel = np.ones((25, 25), 'uint8')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minSize=(10, 10))
    for (x, y, w, h) in faces:
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (90, 180, 100))
        frame[y:y+h, x:x+w, :] = cv2.dilate(frame[y:y+h, x:x+w, :], kernel)
        #Eyes
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255))
        cv2.circle(frame,(int(x+w/3),int(y+h/2.5)),int(h/9),(255, 255, 255),-1)
        cv2.circle(frame,(int(x+w*2/3),int(y+h/2.5)),int(h/9),(255, 255, 255),-1)
        cv2.circle(frame,(int(x+w/3),int(y+h/2.2)),int(h/25),(19, 69, 139),-1)
        cv2.circle(frame,(int(x+w*2/3),int(y+h/2.2)),int(h/25),(19, 69, 169),-1)
        #Eyebrows
        cv2.ellipse(frame,(int(x+w/3),int(y+h/2.5)),(int(h/5),int(h/5)),0,225,305,(19,69,139),5)
        cv2.ellipse(frame,(int(x+w*2/3),int(y+h/2.5)),(int(h/5),int(h/5)),0,225,305,(19,69,139),5)
        #Mouth
        cv2.ellipse(frame,(int(x+w/2),int(y+h/1.6)),(int(w/4),int(h/4)),180,225,315,(0,0,255),5)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
