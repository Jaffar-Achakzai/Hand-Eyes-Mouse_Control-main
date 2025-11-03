import cv2
import mediapipe as mp
import pyautogui as pg

cam  = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

#extract pyautogui frame size
screen_w, screen_h  = pg.size()

while True:
     suc, frame = cam.read()
     frame = cv2.flip(frame, 1)
     frame_h, frame_w, _ = frame.shape
     rgb_fram = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

     result = face_mesh.process(rgb_fram)
     # print("output", output)

     # landmarks_all = result.multi_face_landmarks
     # print(landmarks_all)

     if result.multi_face_landmarks:
          landmarks = result.multi_face_landmarks[0].landmark
          # print(landmarks)

          for i, landmark in enumerate(landmarks[474:748]):

               #convert decimal x and y to integers
               x = int(landmark.x * frame_w)
               y = int(landmark.y * frame_h)
               # print(x , y)

               if i == 1:
                    #scale with relative to frame
                    screen_x = int(screen_w / frame_w * x)
                    screen_y = int(screen_h / frame_h * y)

                    pg.moveTo(screen_x, screen_y)



                    #coloring the landmarks
               cv2.circle(frame, (x,y),3,(0,255,0),-1)


          left = [landmarks[145], landmarks[159]]

          #for just printing left eye
          for landmark in left:
               x = int(landmark.x * frame_w)
               y = int(landmark.y * frame_h)
               cv2.circle(frame, (x,y),3,(0,255,0),-1)

          #left eye blink
          if((left[0].y - left[1].y) < 0.005):
               print("Blink")
               pg.click()
               pg.sleep(1)
          else:
               print((left[0].y - left[1].y))








     cv2.imshow("Camera",frame)

     if cv2.waitKey(1) & 0xFF == ord('q'):
          break