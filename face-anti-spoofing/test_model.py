import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np 
import cv2
#########################
#from keras.models import model_from_yaml
import keras.models as models
from keras.preprocessing.image import img_to_array
#########################
from rPPG.rPPG_Extracter import *
from rPPG.rPPG_lukas_Extracter import *
#########################
import yaml, json
import face_recognition
import mediapipe as mp
import time
import math


# load YAML and create model
yaml_file = open("trained_model/RGB_rPPG_merge_softmax_.yaml", 'r')
loaded_model_yaml = yaml_file.read()
jsonObj = json.dumps(yaml.load(loaded_model_yaml,Loader=yaml.FullLoader))
yaml_file.close()
#model = models.model_from_yaml(loaded_model_yaml)
model = models.model_from_json(jsonObj)
# load weights into new model
model.load_weights("trained_model/RGB_rPPG_merge_softmax_.h5")
print("[INFO] Model is loaded from disk")
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

#손 랜드마크를 이용해 접혔는지 판단(점과 점사이 거리 공식 사용)
def dist(x1, y1, x2, y2):
  return math.sqrt(math.pow(x1 - x2, 2)) + math.sqrt(math.pow(y1 - y2, 2))

#다섯개 손가락 랜드마크(엄지손가락 접힌 길이가 새끼손가락 중간 마디보다 작아야 접힌 걸로 인식)
compareIndex=[[10,4],[6,8],[10,12],[14,16],[18,20]]

open=[False,False,False,False,False]
gesture = [
    [True,True,True,True,True,"in!"],[False,False,False,False,False,'out!']
]


dim = (128,128)
def get_rppg_pred(frame):
    use_classifier = True  # Toggles skin classifier
    use_flow = False       # (Mixed_motion only) Toggles PPG detection with Lukas Kanade optical flow          
    sub_roi = []           # If instead of skin classifier, forhead estimation should be used set to [.35,.65,.05,.15]
    use_resampling = True  # Set to true with webcam
    
    fftlength = 300
    fs = 20
    f = np.arange(0,fs/2,fftlength/2 + 1) * 60;

    timestamps = []
    time_start = [0]

    break_ = False

    rPPG_extracter = rPPG_Extracter()
    rPPG_extracter_lukas = rPPG_Lukas_Extracter()
    bpm = 0
    
    dt = time.time()-time_start[0]
    time_start[0] = time.time()
    if len(timestamps) == 0:
        timestamps.append(0)
    else:
        timestamps.append(timestamps[-1] + dt)
        
    rPPG = []

    rPPG_extracter.measure_rPPG(frame,use_classifier,sub_roi) 
    rPPG = np.transpose(rPPG_extracter.rPPG)
    
        # Extract Pulse
    if rPPG.shape[1] > 10:
        if use_resampling :
            t = np.arange(0,timestamps[-1],1/fs)

            
            rPPG_resampled= np.zeros((3,t.shape[0]))
            for col in [0,1,2]:
                rPPG_resampled[col] = np.interp(t,timestamps,rPPG[col])
            rPPG = rPPG_resampled
        num_frames = rPPG.shape[1]

        t = np.arange(num_frames)/fs
    return rPPG
    

def make_pred(li):
    [single_img,rppg] = li
    single_img = cv2.resize(single_img, dim)
    single_x = img_to_array(single_img)
    single_x = np.expand_dims(single_x, axis=0)
    single_pred = model.predict([single_x,rppg])
    return single_pred


    
cascPath = 'rPPG/util/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

#카메라 선택
video_capture = cv2.VideoCapture(1,cv2.CAP_DSHOW)

obama_image = face_recognition.load_image_file("obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

biden_image = face_recognition.load_image_file("biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

moon_image = face_recognition.load_image_file("moon.jpg")
moon_face_encoding = face_recognition.face_encodings(moon_image)[0]

an_image = face_recognition.load_image_file("an.jpg")
an_face_encoding = face_recognition.face_encodings(an_image)[0]

known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding,
    moon_face_encoding,
    an_face_encoding
]
known_face_names = [
    "Barack Obama",
    "Joe Biden",
    "Moon",
    "an"
]
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

collected_results = []
counter = 0          # count collected buffers
frames_buffer = 5    # how many frames to collect to check for
accepted_falses = 1  # how many should have zeros to say it is real
name=""
with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.8) as hands:
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        rgb_small_frame = small_frame[:, :, ::-1]
        if process_this_frame:

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:

                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # # known_face_encodings에서 일치하는 항목이 발견되면 첫 번째 항목만 사용합니다.
                # 일치하는 경우 True인 경우:
                # first_match_index = match.index(True)
                # name = known_face_names[first_match_index]

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.15,
                minNeighbors=6,
                minSize=(40, 40)
            )

            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                sub_img=frame[y:y+h,x:x+w]
                rppg_s = get_rppg_pred(sub_img)
                rppg_s = rppg_s.T

                pred = make_pred([sub_img,rppg_s])

                collected_results.append(np.argmax(pred))
                counter += 1
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame,"Real: "+str(pred[0][0]), (50,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
                cv2.putText(frame,"Fake: "+str(pred[0][1]), (50,60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
                if len(collected_results) == frames_buffer:
                    print(sum(collected_results))
                    if sum(collected_results) <= 0.75:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, name, (x+w + 6, y+h - 6), font, 1.0, (255, 255, 255), 1)
                    else:
                        name="Fake"
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(frame, name, (x + w + 6, y + h - 6), font, 1.0, (255, 255, 255), 1)
                    collected_results.pop(0)
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        frame.flags.writeable = False
        results = hands.process(frame)
        image_height, image_width, _ = frame.shape
        # Draw the hand annotations on the image.
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        a = 0.0
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Here is How to Get All the Coordinates
                for ids, landmrk in enumerate(hand_landmarks.landmark):
                    # print(ids, landmrk)
                    cx, cy = landmrk.x * image_width, landmrk.y * image_height
                    if (a == 0.0):
                        a = cx

                    # print(cx, a)

                    if (cx - a >= 87):
                        a = 0.0
                        print('left')
                    elif (cx - a <= -110):
                        a = 0.0
                        print('right')

                    else:
                        a = cx
                    # 손가락 접히는거 구분
                    for i in range(0, 5):
                        open[i] = dist(hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y,
                                       hand_landmarks.landmark[compareIndex[i][0]].x,
                                       hand_landmarks.landmark[compareIndex[i][0]].y) < dist(
                            hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y,
                            hand_landmarks.landmark[compareIndex[i][1]].x,
                            hand_landmarks.landmark[compareIndex[i][1]].y)
                    # print(open)
                    for i in range(0, len(gesture)):
                        flag = True
                        for j in range(0, 5):
                            if (gesture[i][j] != open[j]):
                                flag = False
                            # hi만 나오게 조건
                        if (flag == True):
                            print(gesture[i][5])

                    # time.sleep(0.2)
                    # print (ids, cx, cy)
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)


        # Display the resulting frame
        cv2.imshow('To quit press q', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
