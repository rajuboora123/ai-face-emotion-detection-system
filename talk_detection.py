import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)

talk_count = 0
talk_state = False

# Audio detection function
def get_audio_level():
    duration = 0.15
    fs = 16000
    audio = sd.rec(int(duration*fs),
                   samplerate=fs,
                   channels=1,
                   blocking=True)
    return np.linalg.norm(audio)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h,w,_ = frame.shape
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results=face_mesh.process(rgb)

    audio_level = get_audio_level()
    mouth_open=False

    if results.multi_face_landmarks:
        lm=results.multi_face_landmarks[0].landmark

        face_width=abs(lm[234].x-lm[454].x)
        mouth_h=abs(lm[13].y-lm[14].y)/face_width

        if mouth_h>0.15:
            mouth_open=True

    # Final talking detection
    if mouth_open and audio_level>8:
        if not talk_state:
            talk_count+=1
            talk_state=True
    else:
        talk_state=False

    cv2.putText(frame,f"Talking Count: {talk_count}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

    cv2.putText(frame,f"Audio Level: {int(audio_level)}",
                (20,80),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow("Audio+Video Talking Detection",frame)

    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()