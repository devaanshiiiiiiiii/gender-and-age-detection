import cv2
from pathlib import Path

def facebox(facenet,fram):
    frameheight=frame.shape[0]
    framewidth=frame.shape[1]

    blob=cv2.dnn.blobFromImage(frame,1.0,(227,227),[104,117,123],swapRB=False)
    facenet.setInput(blob)
    detection=facenet.forward()
    bbox=[]
    for i in range(detection.shape[2]):
        confidance=detection[0,0,i,2]
        if confidance>0.7:
            x1=int(detection[0,0,i,3]*framewidth)
            y1=int(detection[0,0,i,4]*frameheight)
            x2=int(detection[0,0,i,5]*framewidth)
            y2=int(detection[0,0,i,6]*frameheight)
            bbox.append([x1,y1,x2,y2])
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
    return frame ,bbox

video=cv2.VideoCapture(0)

faceproto=Path(r"C:\Users\ASUS\OneDrive\Desktop\parent folder\age aand gender detection\opencv_face_detector.pbtxt")
facemodel=Path(r"C:\Users\ASUS\OneDrive\Desktop\parent folder\age aand gender detection\opencv_face_detector_uint8.pb")

ageproto=Path(r"C:\Users\ASUS\OneDrive\Desktop\parent folder\age aand gender detection\age_deploy.prototxt")
agemodel=Path(r"C:\Users\ASUS\OneDrive\Desktop\parent folder\age aand gender detection\age_net.caffemodel")

genderproto=Path(r"C:\Users\ASUS\OneDrive\Desktop\parent folder\age aand gender detection\gender_deploy.prototxt")
gendermodel=Path(r"C:\Users\ASUS\OneDrive\Desktop\parent folder\age aand gender detection\gender_net.caffemodel")

facenet=cv2.dnn.readNet(facemodel,faceproto)
agenet=cv2.dnn.readNet(agemodel,ageproto)
gender=cv2.dnn.readNet(gendermodel,genderproto)

agelist=['(0-2)','(4-6)','(8-12)','(15-20)','(25-32)','(38-43)','(48-53)','(60-100)']
genderlist=['male','female']
model_mean_value=(78.4263377603,87.7689143744,114.895847746)


while True:
    ret,frame=video.read()
    framenet,bbox=facebox(facenet, frame)
    for _ in bbox:
        face=framenet[_[1]:_[3], _[0]:_[2]]
        blob=cv2.dnn.blobFromImage(face,1.0,(227,227),model_mean_value,swapRB=False)
        gender.setInput(blob)
        genderpredict=gender.forward()
        gen=genderlist[genderpredict[0].argmax()]

        agenet.setInput(blob)
        agepredict=agenet.forward()
        age=agelist[agepredict[0].argmax()]

        lable="{},{}".format(gen,age)
        cv2.putText(frame,lable,(_[0],_[1]-10), cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2,cv2.LINE_AA)
    cv2.imshow("age-gender",framenet)
    k=cv2.waitKey(1)
    if k==ord("q"):
        break
video.release()
cv2.destroyAllWindows()
