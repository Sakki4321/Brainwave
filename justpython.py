import cv2
import mediapipe as mp

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

def goto():
    exercise_input = int(input(
        "choose your exerice- \n 1)Push Up \n 2)Pull up \n 3)Bicep Curls \n 4)Shoulder raises(any type) \n 5)Squats \n enter a valid number from above"))
    #if exercise_input == 1:
    # elif exercise_input == 2:
    # elif exercise_input == 3:
    if exercise_input == 4:
        cap = cv2.VideoCapture(0)
        Pos =""
        UP = False
        counter = 0
        while True:
            success, img = cap.read()
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)
            if results.pose_landmarks:
                mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
                points = {}
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # print(id,lm,cx,cy)
                    points[id] = (cx, cy)

                cv2.circle(img, points[12], 15, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, points[14], 15, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, points[11], 15, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, points[13], 15, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, points[15], 15, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, points[16], 15, (255, 0, 0), cv2.FILLED)

                if not UP and points[14][1] + 30 < points[11][1]:
                    pos = "UP"
                    print(pos)
                    UP = True
                    counter += 1
                    print(counter,"\n------------------\n")
                elif UP and points[14][1] + 30 > points[11][1]:
                    pos = "DOWN"
                    print(pos)
                    UP = False
                elif points[15][0]  < points[16][0]:
                    print("Exit\n Thanks for using this program")
                    return
                elif points[11][0] < points[16][0]:
                    goto()
            cv2.putText(img, str(counter), (100, 150), cv2.FONT_HERSHEY_PLAIN, 9, (255, 255, 255))
            cv2.imshow('img', img)
            cv2.waitKey(1)
   #elif exercise_input == 5:

    else:
        print("wrong input")
        ans = int(input("If you want to quit press 1 \n If you want to continue press 0"))
        if ans == 1:
            print("Thanks for using this program")
            return
        elif ans == 0:
            goto()
goto()






