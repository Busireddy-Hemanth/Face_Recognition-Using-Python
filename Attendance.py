import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
path="image_attendance"
images=[]
detect=[]
classname=[]
mylist=os.listdir(path)
print(mylist)
for cl in mylist:
    curImg=cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classname.append(os.path.splitext(cl)[0])
def findEncodings(images):
    encodeList=[]
    for img in images:
         img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
         encodings=face_recognition.face_encodings(img)[0]
         encodeList.append(encodings)
    return encodeList
encodelistknown=findEncodings(images)
print("Encodings Complete")
cam=cv2.VideoCapture(0)
while(True):
    success,image=cam.read()
    font = cv2.FONT_HERSHEY_SIMPLEX 
    if success==True:
        cv2.imshow('Live',image)
        cv2.putText(image,"hiii",50,40,font,'red',3)
    imgs=cv2.resize(image,(0,0),None,0.25,0.25)
    face_names=[]
    imgs=cv2.cvtColor(image,cv2.COLOR_BGR2RGB) 
    facesCurFrame=face_recognition.face_locations(imgs)
    encodeCurFrame=face_recognition.face_encodings(imgs,facesCurFrame)
    for encodeface in encodeCurFrame:
        matches=face_recognition.compare_faces(encodelistknown,encodeface)
        facedis=face_recognition.face_distance(encodelistknown,encodeface)
        name="unknown"
        print(np.argmin(facedis))
        best_match_index = np.argmin(facedis)
        print(best_match_index)
        print(matches[best_match_index])
        if matches[best_match_index]:
                name = classname[best_match_index]

        face_names.append(name)
        print(face_names)
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
cam.release()
cv2.destroyAllWindows()
    








        
        
