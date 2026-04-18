import warnings
warnings.filterwarnings("ignore")

import cv2
import mediapipe as mp
import numpy as np
from collections import deque

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)

LEFT_EYE=[33,160,158,133,153,144]
RIGHT_EYE=[362,385,387,263,373,380]

blink_count=0
talk_count=0

blink_frames=0
talk_frames=0

emotion_buffer=deque(maxlen=20)
mouth_history=deque(maxlen=10)

neutral_mouth=None

def dist(a,b):
    return np.linalg.norm(np.array(a)-np.array(b))

def eye_ratio(lm,idx,w,h):
    pts=[(int(lm[i].x*w),int(lm[i].y*h)) for i in idx]
    return (dist(pts[1],pts[5])+dist(pts[2],pts[4]))/(2*dist(pts[0],pts[3]))

while True:
    ret,frame=cap.read()
    if not ret:
        break

    h,w,_=frame.shape
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results=face_mesh.process(rgb)

    emotion="Neutral"

    if results.multi_face_landmarks:
        lm=results.multi_face_landmarks[0].landmark

        face_width=abs(lm[234].x-lm[454].x)
        if face_width==0:
            continue

        # BLINK DETECTION
        ear=(eye_ratio(lm,LEFT_EYE,w,h)+eye_ratio(lm,RIGHT_EYE,w,h))/2
        if ear<0.19:
            blink_frames+=1
        else:
            if blink_frames>=3:
                blink_count+=1
            blink_frames=0

        # TALKING DETECTION
        mouth_h=abs(lm[13].y-lm[14].y)/face_width
        if neutral_mouth is None:
            neutral_mouth=mouth_h

        mouth_history.append(mouth_h)
        avg_mouth=sum(mouth_history)/len(mouth_history)

        if avg_mouth>neutral_mouth+0.07:
            talk_frames+=1
        else:
            if talk_frames>=5:
                talk_count+=1
            talk_frames=0

        # EMOTION DETECTION
        mouth_w=abs(lm[78].x-lm[308].x)/face_width
        eyebrow=abs(lm[70].y-lm[159].y)/face_width

        if mouth_w>0.50:
            emotion="Happy"

        elif avg_mouth>neutral_mouth+0.12:
            emotion="Surprise"

        elif lm[61].y>lm[13].y and lm[291].y>lm[13].y:
            emotion="Sad"

        elif eyebrow<0.06:
            emotion="Angry"

        elif 0.06<eyebrow<0.09:
            emotion="Confused"

        emotion_buffer.append(emotion)
        emotion=max(set(emotion_buffer),
                    key=emotion_buffer.count)

    cv2.putText(frame,f"Blinks: {blink_count}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

    cv2.putText(frame,f"Talking: {talk_count}",
                (20,80),
                cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

    cv2.putText(frame,f"Emotion: {emotion}",
                (20,120),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow("AI Face Emotion System",frame)

    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()
