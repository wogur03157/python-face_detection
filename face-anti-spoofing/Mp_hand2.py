import cv2
import mediapipe as mp
import time
import math

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

# For webcam input:
cap = cv2.VideoCapture(1)
with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.8) as hands:
  while cap.isOpened():
    success, frame = cap.read()
    if not success:
      #print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

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
            cx, cy = landmrk.x * image_width, landmrk.y*image_height
            if (a == 0.0) :
                a = cx

           # print(cx, a)

            if (cx-a>=87) :
                a = 0.0
                print('left')
            elif (cx-a<=-110) :
                a = 0.0
                print('right')

            else :
                a = cx
            #손가락 접히는거 구분
            for i in range(0,5):
              open[i] = dist(hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y,
                                    hand_landmarks.landmark[compareIndex[i][0]].x,hand_landmarks.landmark[compareIndex[i][0]].y) < dist(hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y,
                                                                                                                                        hand_landmarks.landmark[compareIndex[i][1]].x, hand_landmarks.landmark[compareIndex[i][1]].y)
            #print(open)
            for i in range(0,len(gesture)):
              flag= True
              for j in range(0,5):
                if(gesture[i][j] != open[j]):
                  flag = False
                # hi만 나오게 조건
              if(flag == True):
                print(gesture[i][5])

            #time.sleep(0.2)
            #print (ids, cx, cy)
        mp_drawing.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    cv2.imshow('MediaPipe Hands', frame)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()

