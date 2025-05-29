import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

from absl import logging
logging.use_absl_handler()

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mp_holistic = mp.solutions.holistic

def get_point(landmark, image_shape):
    return (int(landmark.x * image_shape[1]), int(landmark.y * image_shape[0]))

def draw_tracker(img, current_position_ratio):
    """
    Draws a horizontal progress tracker bar on the image at the top-left corner.
    current_position_ratio: a value between 0 and 1 representing the current completion level (0=down, 1=up).
    """
    h, w, _ = img.shape
    bar_length = 400  # Length of the tracker bar
    bar_x_start, bar_y = 50, 50  # Position at the top-left corner
    bar_x_end = bar_x_start + bar_length

    # Calculate the current filled length of the bar
    fill_x_end = int(bar_x_start + (bar_length * current_position_ratio))

    # Draw the background of the bar
    cv2.rectangle(img, (bar_x_start, bar_y - 20), (bar_x_end, bar_y), (200, 200, 200), -1)
    # Draw the filled portion of the bar
    cv2.rectangle(img, (bar_x_start, bar_y - 20), (fill_x_end, bar_y), (0, 255, 0), -1)

def PushUp():
    cap = cv2.VideoCapture(0)
    pos = ""
    UP = False
    counter = 0
    current_position_ratio = 0.0
    step_size = 0.05  # Controls the speed of the transition
    target_position_ratio = 0.0
    show_arrows = False
    arrow_direction = "UP"
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            success, img = cap.read()
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = holistic.process(imgRGB)
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
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
                wrist_width = 10  # Default thickness in case landmarks are missing
                if results.right_hand_landmarks:
                    right_hand_landmarks = results.right_hand_landmarks.landmark
                    wrist_coord = get_point(right_hand_landmarks[mp_holistic.HandLandmark.WRIST], img.shape)
                    pinky_coord = get_point(right_hand_landmarks[mp_holistic.HandLandmark.PINKY_MCP], img.shape)
                    index_coord = get_point(right_hand_landmarks[mp_holistic.HandLandmark.INDEX_FINGER_MCP], img.shape)
                    wrist_width = int(np.linalg.norm(np.array(pinky_coord) - np.array(index_coord)))

                # Draw arms if landmarks are available
                if landmarks is not None:  # Ensure landmarks are defined
                    for side in ['RIGHT', 'LEFT']:
                        shoulder_pt = get_point(landmarks[getattr(mp_holistic.PoseLandmark, f'{side}_SHOULDER')], img.shape)
                        elbow_pt = get_point(landmarks[getattr(mp_holistic.PoseLandmark, f'{side}_ELBOW')], img.shape)
                        wrist_pt = get_point(landmarks[getattr(mp_holistic.PoseLandmark, f'{side}_WRIST')], img.shape)

                        arm_points = np.array([shoulder_pt, elbow_pt, wrist_pt], np.int32).reshape((-1, 1, 2))
                        overlay = img.copy()
                        cv2.polylines(overlay, [arm_points], isClosed=False, color=(0, 255, 0), thickness=wrist_width)
                        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

                right_shoulder_pt = get_point(landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER], img.shape)
                right_elbow_pt = get_point(landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW], img.shape)
                right_wrist_pt = get_point(landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST], img.shape)

                right_arm_points = np.array([right_shoulder_pt, right_elbow_pt, right_wrist_pt], np.int32)
                right_arm_points = right_arm_points.reshape((-1, 1, 2))

                # Define and draw the left arm (shoulder, elbow, wrist)
                left_shoulder_pt = get_point(landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER], img.shape)
                left_elbow_pt = get_point(landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW], img.shape)
                left_wrist_pt = get_point(landmarks[mp_holistic.PoseLandmark.LEFT_WRIST], img.shape)

                left_arm_points = np.array([left_shoulder_pt, left_elbow_pt, left_wrist_pt], np.int32)
                left_arm_points = left_arm_points.reshape((-1, 1, 2))

                # Define chest area points (shoulders and hips)
                right_hip_pt = get_point(landmarks[mp_holistic.PoseLandmark.RIGHT_HIP], img.shape)
                left_hip_pt = get_point(landmarks[mp_holistic.PoseLandmark.LEFT_HIP], img.shape)

                # Define chest points using an array of tuples for fillPoly
                chest_points = np.array([
                    left_shoulder_pt,
                    right_shoulder_pt,
                    [(right_shoulder_pt[0]+right_hip_pt[0])/2,(right_shoulder_pt[1]+right_hip_pt[1])/2],
                    [(left_shoulder_pt[0]+left_hip_pt[0])/2,(left_shoulder_pt[1]+left_hip_pt[1])/2]
                ], np.int32)

                # Create an overlay and add translucent polygons for both arms and chest area
                overlay = img.copy()

                # Draw arms
                cv2.polylines(overlay, [right_arm_points], isClosed=False, color=(0, 255, 0), thickness=10)
                cv2.polylines(overlay, [left_arm_points], isClosed=False, color=(0, 255, 0), thickness=10)

                # Draw chest area
                cv2.fillPoly(overlay, [chest_points], color=(0, 0, 255))  # Fill the chest area with red
                cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

                if not UP and points[12][1] < points[14][1]:
                    pos = "UP"
                    print(pos)
                    UP = True
                    counter += 1
                    target_position_ratio = 1.0  # UP position
                    arrow_direction = "UP"
                    show_arrows = True
                    print(counter, "\n------------------\n")
                elif UP and points[12][1] - 60 > points[14][1]:
                    pos = "DOWN"
                    print(pos)
                    target_position_ratio = 0.0 # DOWN position
                    arrow_direction = "DOWN"
                    show_arrows = True
                    UP = False
                elif points[15][0] < points[16][0]:
                    print("Exit\nThanks for using this program")
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                elif points[11][0] < points[16][0]:
                    goto()
                if show_arrows:
                    for pt in [12, 11]:
                        if arrow_direction == "UP":
                            cv2.arrowedLine(img, (points[pt][0], points[pt][1] - 50), points[pt], (0, 0, 255), 3,
                                            cv2.LINE_AA, tipLength=0.4)
                        elif arrow_direction == "DOWN":
                            cv2.arrowedLine(img, (points[pt][0], points[pt][1] + 50), points[pt], (0, 0, 255), 3,
                                            cv2.LINE_AA, tipLength=0.4)
                # Smoothly update current_position_ratio toward target_position_ratio
                if current_position_ratio < target_position_ratio:
                    current_position_ratio = min(current_position_ratio + step_size, target_position_ratio)
                elif current_position_ratio > target_position_ratio:
                    current_position_ratio = max(current_position_ratio - step_size, target_position_ratio)
                draw_tracker(img, current_position_ratio)
                # Reset `show_arrows` after the transition to make arrows persist
                if current_position_ratio == target_position_ratio:
                    show_arrows = False
            cv2.putText(img, str(counter) + "  " + pos, (100, 150), cv2.FONT_HERSHEY_PLAIN, 6, (0, 0, 255))
            cv2.imshow('img', img)
            cv2.waitKey(1)
def PullUp():
    cap = cv2.VideoCapture(0)
    pos = ""
    UP = False
    counter = 0
    current_position_ratio = 0.0
    step_size = 0.05  # Controls the speed of the transition
    target_position_ratio = 0.0
    show_arrows = False
    arrow_direction = "UP"
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            success, img = cap.read()
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = holistic.process(imgRGB)
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
                points = {}
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # print(id,lm,cx,cy)
                    points[id] = (cx, cy)

                cv2.circle(img, points[12], 15, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, points[16], 15, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, points[11], 15, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, points[15], 15, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, points[16], 15, (255, 0, 0), cv2.FILLED)
                wrist_width = 10  # Default thickness in case landmarks are missing
                if results.right_hand_landmarks:
                    right_hand_landmarks = results.right_hand_landmarks.landmark
                    wrist_coord = get_point(right_hand_landmarks[mp_holistic.HandLandmark.WRIST], img.shape)
                    pinky_coord = get_point(right_hand_landmarks[mp_holistic.HandLandmark.PINKY_MCP], img.shape)
                    index_coord = get_point(right_hand_landmarks[mp_holistic.HandLandmark.INDEX_FINGER_MCP], img.shape)
                    wrist_width = int(np.linalg.norm(np.array(pinky_coord) - np.array(index_coord)))

                # Draw arms if landmarks are available
                if landmarks is not None:  # Ensure landmarks are defined
                    for side in ['RIGHT', 'LEFT']:
                        shoulder_pt = get_point(landmarks[getattr(mp_holistic.PoseLandmark, f'{side}_SHOULDER')], img.shape)
                        elbow_pt = get_point(landmarks[getattr(mp_holistic.PoseLandmark, f'{side}_ELBOW')], img.shape)
                        wrist_pt = get_point(landmarks[getattr(mp_holistic.PoseLandmark, f'{side}_WRIST')], img.shape)

                        arm_points = np.array([shoulder_pt, elbow_pt, wrist_pt], np.int32).reshape((-1, 1, 2))
                        overlay = img.copy()
                        cv2.polylines(overlay, [arm_points], isClosed=False, color=(0, 255, 0), thickness=wrist_width)
                        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

                right_shoulder_pt = get_point(landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER], img.shape)
                right_elbow_pt = get_point(landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW], img.shape)
                right_wrist_pt = get_point(landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST], img.shape)

                right_arm_points = np.array([right_shoulder_pt, right_elbow_pt, right_wrist_pt], np.int32)
                right_arm_points = right_arm_points.reshape((-1, 1, 2))

                # Define and draw the left arm (shoulder, elbow, wrist)
                left_shoulder_pt = get_point(landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER], img.shape)
                left_elbow_pt = get_point(landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW], img.shape)
                left_wrist_pt = get_point(landmarks[mp_holistic.PoseLandmark.LEFT_WRIST], img.shape)

                left_arm_points = np.array([left_shoulder_pt, left_elbow_pt, left_wrist_pt], np.int32)
                left_arm_points = left_arm_points.reshape((-1, 1, 2))
                # Create an overlay and add translucent polygons for both arms and chest area
                overlay = img.copy()

                # Draw arms
                cv2.polylines(overlay, [right_arm_points], isClosed=False, color=(0, 255, 0), thickness=10)
                cv2.polylines(overlay, [left_arm_points], isClosed=False, color=(0, 255, 0), thickness=10)

                if not UP and (points[12][1] - 50 < points[16][1] and points[11][1] - 50 < points[15][1]):
                    pos = "UP"
                    print(pos)
                    UP = True
                    counter += 1
                    target_position_ratio = 1.0  # UP position
                    arrow_direction = "UP"
                    show_arrows = True
                    print(counter, "\n------------------\n")
                elif UP and (points[12][1] - 200 > points[16][1] and points[11][1] - 200 > points[15][1]):
                    pos = "DOWN"
                    print(pos)
                    target_position_ratio = 0.0  # DOWN position
                    arrow_direction = "DOWN"
                    show_arrows = True
                    UP = False

                elif points[15][0] < points[16][0]:
                    print("Exit\nThanks for using this program")
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                elif points[11][0] < points[16][0]:
                    goto()
                if show_arrows:
                    for pt in [13, 14]:
                        if arrow_direction == "DOWN":
                            cv2.arrowedLine(img, (points[pt][0], points[pt][1] - 50), points[pt], (0, 0, 255), 3,
                                            cv2.LINE_AA, tipLength=0.4)
                        elif arrow_direction == "UP":
                            cv2.arrowedLine(img, (points[pt][0], points[pt][1] + 50), points[pt], (0, 0, 255), 3,
                                            cv2.LINE_AA, tipLength=0.4)
                # Smoothly update current_position_ratio toward target_position_ratio
                if current_position_ratio < target_position_ratio:
                    current_position_ratio = min(current_position_ratio + step_size, target_position_ratio)
                elif current_position_ratio > target_position_ratio:
                    current_position_ratio = max(current_position_ratio - step_size, target_position_ratio)
                draw_tracker(img, current_position_ratio)
                # Reset `show_arrows` after the transition to make arrows persist
                if current_position_ratio == target_position_ratio:
                    show_arrows = False

            cv2.putText(img, str(counter) + "  " + pos, (100, 150), cv2.FONT_HERSHEY_PLAIN, 6, (0, 0, 255))
            cv2.imshow('img', img)
            cv2.waitKey(1)
def Curls():
    cap = cv2.VideoCapture(0)
    pos = ""
    UP = False
    counter = 0
    current_position_ratio = 0.0
    step_size = 0.05  # Controls the speed of the transition
    target_position_ratio = 0.0
    show_arrows = False
    arrow_direction = "UP"
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            success, img = cap.read()
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = holistic.process(imgRGB)
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS )
                points = {}
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # print(id,lm,cx,cy)
                    points[id] = (cx, cy)

                cv2.circle(img, points[16], 15, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, points[14], 15, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, points[15], 15, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, points[13], 15, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, points[15], 15, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, points[16], 15, (255, 0, 0), cv2.FILLED)
                wrist_width = 10  # Default thickness in case landmarks are missing
                if results.right_hand_landmarks:
                    right_hand_landmarks = results.right_hand_landmarks.landmark
                    wrist_coord = get_point(right_hand_landmarks[mp_holistic.HandLandmark.WRIST], img.shape)
                    pinky_coord = get_point(right_hand_landmarks[mp_holistic.HandLandmark.PINKY_MCP], img.shape)
                    index_coord = get_point(right_hand_landmarks[mp_holistic.HandLandmark.INDEX_FINGER_MCP], img.shape)
                    wrist_width = int(np.linalg.norm(np.array(pinky_coord) - np.array(index_coord)))

                # Draw arms if landmarks are available
                if landmarks is not None:  # Ensure landmarks are defined
                    for side in ['RIGHT', 'LEFT']:
                        shoulder_pt = get_point(landmarks[getattr(mp_holistic.PoseLandmark, f'{side}_SHOULDER')], img.shape)
                        elbow_pt = get_point(landmarks[getattr(mp_holistic.PoseLandmark, f'{side}_ELBOW')], img.shape)
                        wrist_pt = get_point(landmarks[getattr(mp_holistic.PoseLandmark, f'{side}_WRIST')], img.shape)

                        arm_points = np.array([shoulder_pt, elbow_pt, wrist_pt], np.int32).reshape((-1, 1, 2))
                        overlay = img.copy()
                        cv2.polylines(overlay, [arm_points], isClosed=False, color=(0, 255, 0), thickness=wrist_width)
                        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

                right_shoulder_pt = get_point(landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER], img.shape)
                right_elbow_pt = get_point(landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW], img.shape)
                right_wrist_pt = get_point(landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST], img.shape)

                right_arm_points = np.array([right_shoulder_pt, right_elbow_pt, right_wrist_pt], np.int32)
                right_arm_points = right_arm_points.reshape((-1, 1, 2))

                # Define and draw the left arm (shoulder, elbow, wrist)
                left_shoulder_pt = get_point(landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER], img.shape)
                left_elbow_pt = get_point(landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW], img.shape)
                left_wrist_pt = get_point(landmarks[mp_holistic.PoseLandmark.LEFT_WRIST], img.shape)

                left_arm_points = np.array([left_shoulder_pt, left_elbow_pt, left_wrist_pt], np.int32)
                left_arm_points = left_arm_points.reshape((-1, 1, 2))
                # Create an overlay and add translucent polygons for both arms and chest area
                overlay = img.copy()

                # Draw arms
                cv2.polylines(overlay, [right_arm_points], isClosed=False, color=(0, 255, 0), thickness=10)
                cv2.polylines(overlay, [left_arm_points], isClosed=False, color=(0, 255, 0), thickness=10)


                if not UP and (points[14][1] - 50 > points[16][1] and points[13][1] - 50 > points[15][1]):
                    pos = "UP"
                    print(pos)
                    UP = True
                    counter += 1
                    target_position_ratio = 1.0  # UP position
                    arrow_direction = "UP"
                    show_arrows = True
                    print(counter, "\n------------------\n")
                elif UP and (points[14][1] + 70 < points[16][1] and points[15][1] + 70 > points[13][1]):
                    pos = "DOWN"
                    print(pos)
                    target_position_ratio = 0.0  # DOWN position
                    arrow_direction = "DOWN"
                    show_arrows = True
                    UP = False
                elif points[15][0] < points[16][0]:
                    print("Exit\nThanks for using this program")
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                elif points[11][0] < points[16][0]:
                    goto()
                if show_arrows:
                    for pt in [15, 16]:
                        if arrow_direction == "UP":
                            cv2.arrowedLine(img, (points[pt][0], points[pt][1] - 50), points[pt], (0, 0, 255), 3,
                                            cv2.LINE_AA, tipLength=0.4)
                        elif arrow_direction == "DOWN":
                            cv2.arrowedLine(img, (points[pt][0], points[pt][1] + 50), points[pt], (0, 0, 255), 3,
                                            cv2.LINE_AA, tipLength=0.4)
                # Smoothly update current_position_ratio toward target_position_ratio
                if current_position_ratio < target_position_ratio:
                     current_position_ratio = min(current_position_ratio + step_size, target_position_ratio)
                elif current_position_ratio > target_position_ratio:
                     current_position_ratio = max(current_position_ratio - step_size, target_position_ratio)
                draw_tracker(img, current_position_ratio)
                # Reset `show_arrows` after the transition to make arrows persist
                if current_position_ratio == target_position_ratio:
                    show_arrows = False
            cv2.putText(img, str(counter) + "  " + pos, (100, 150), cv2.FONT_HERSHEY_PLAIN, 6, (0, 0, 255))
            cv2.imshow('img', img)
            cv2.waitKey(1)
def Shoulder_raises():
    cap = cv2.VideoCapture(0)
    pos = ""
    UP = False
    counter = 0
    current_position_ratio = 0.0
    step_size = 0.05  # Controls the speed of the transition
    target_position_ratio = 0.0
    show_arrows = False
    arrow_direction = "UP"
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            success, img = cap.read()
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = holistic.process(imgRGB)
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
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
                wrist_width = 10  # Default thickness in case landmarks are missing
                if results.right_hand_landmarks:
                    right_hand_landmarks = results.right_hand_landmarks.landmark
                    wrist_coord = get_point(right_hand_landmarks[mp_holistic.HandLandmark.WRIST], img.shape)
                    pinky_coord = get_point(right_hand_landmarks[mp_holistic.HandLandmark.PINKY_MCP], img.shape)
                    index_coord = get_point(right_hand_landmarks[mp_holistic.HandLandmark.INDEX_FINGER_MCP], img.shape)
                    wrist_width = int(np.linalg.norm(np.array(pinky_coord) - np.array(index_coord)))

                # Draw arms if landmarks are available
                if landmarks is not None:  # Ensure landmarks are defined
                    for side in ['RIGHT', 'LEFT']:
                        shoulder_pt = get_point(landmarks[getattr(mp_holistic.PoseLandmark, f'{side}_SHOULDER')], img.shape)
                        elbow_pt = get_point(landmarks[getattr(mp_holistic.PoseLandmark, f'{side}_ELBOW')], img.shape)
                        wrist_pt = get_point(landmarks[getattr(mp_holistic.PoseLandmark, f'{side}_WRIST')], img.shape)

                        arm_points = np.array([shoulder_pt, elbow_pt, wrist_pt], np.int32).reshape((-1, 1, 2))
                        overlay = img.copy()
                        cv2.polylines(overlay, [arm_points], isClosed=False, color=(0, 255, 0), thickness=wrist_width)
                        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

                right_shoulder_pt = get_point(landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER], img.shape)
                right_elbow_pt = get_point(landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW], img.shape)
                right_wrist_pt = get_point(landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST], img.shape)

                right_arm_points = np.array([right_shoulder_pt, right_elbow_pt, right_wrist_pt], np.int32)
                right_arm_points = right_arm_points.reshape((-1, 1, 2))

                # Define and draw the left arm (shoulder, elbow, wrist)
                left_shoulder_pt = get_point(landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER], img.shape)
                left_elbow_pt = get_point(landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW], img.shape)
                left_wrist_pt = get_point(landmarks[mp_holistic.PoseLandmark.LEFT_WRIST], img.shape)

                left_arm_points = np.array([left_shoulder_pt, left_elbow_pt, left_wrist_pt], np.int32)
                left_arm_points = left_arm_points.reshape((-1, 1, 2))
                # Create an overlay and add translucent polygons for both arms and chest area
                overlay = img.copy()

                # Draw arms
                cv2.polylines(overlay, [right_arm_points], isClosed=False, color=(0, 255, 0), thickness=10)
                cv2.polylines(overlay, [left_arm_points], isClosed=False, color=(0, 255, 0), thickness=10)

                if not UP and (points[14][1] + 30 < points[12][1] and points[13][1] + 30 < points[11][1]) :
                    pos = "UP"
                    print(pos)
                    UP = True
                    counter += 1
                    target_position_ratio = 1.0  # UP position
                    arrow_direction = "UP"
                    show_arrows = True
                    print(counter, "\n------------------\n")
                elif UP and (points[14][1] - 30 > points[12][1] and points[13][1] - 30 > points[11][1]):
                    pos = "DOWN"
                    print(pos)
                    target_position_ratio = 0.0  # DOWN position
                    arrow_direction = "DOWN"
                    show_arrows = True
                    UP = False
                elif points[15][0] < points[16][0]:
                    print("Exit\nThanks for using this program")
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                elif points[11][0] < points[16][0]:
                    goto()
                if show_arrows:
                    for pt in [13, 14]:
                        if arrow_direction == "UP":
                            cv2.arrowedLine(img, (points[pt][0], points[pt][1] - 50), points[pt], (0, 0, 255), 3,
                                            cv2.LINE_AA, tipLength=0.4)
                        elif arrow_direction == "DOWN":
                            cv2.arrowedLine(img, (points[pt][0], points[pt][1] + 50), points[pt], (0, 0, 255), 3,
                                            cv2.LINE_AA, tipLength=0.4)
                # Smoothly update current_position_ratio toward target_position_ratio
                if current_position_ratio < target_position_ratio:
                    current_position_ratio = min(current_position_ratio + step_size, target_position_ratio)
                elif current_position_ratio > target_position_ratio:
                    current_position_ratio = max(current_position_ratio - step_size, target_position_ratio)
                draw_tracker(img, current_position_ratio)
                # Reset `show_arrows` after the transition to make arrows persist
                if current_position_ratio == target_position_ratio:
                    show_arrows = False
            cv2.putText(img, str(counter) + "  " + pos, (100, 150), cv2.FONT_HERSHEY_PLAIN, 6, (0, 0, 255))
            cv2.imshow('img', img)
            cv2.waitKey(1)
def Squats():
    cap = cv2.VideoCapture(0)
    pos = ""
    UP = False
    counter = 0
    current_position_ratio = 0.0
    step_size = 0.05  # Controls the speed of the transition
    target_position_ratio = 0.0
    show_arrows = False
    arrow_direction = "UP"
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            success, img = cap.read()
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = holistic.process(imgRGB)
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
                points = {}
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # print(id,lm,cx,cy)
                    points[id] = (cx, cy)

                cv2.circle(img, points[24], 15, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, points[26], 15, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, points[23], 15, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, points[25], 15, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, points[15], 15, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, points[16], 15, (255, 0, 0), cv2.FILLED)
                wrist_width = 10  # Default thickness in case landmarks are missing
                if results.right_hand_landmarks:
                    right_hand_landmarks = results.right_hand_landmarks.landmark
                    wrist_coord = get_point(right_hand_landmarks[mp_holistic.HandLandmark.WRIST], img.shape)
                    pinky_coord = get_point(right_hand_landmarks[mp_holistic.HandLandmark.PINKY_MCP], img.shape)
                    index_coord = get_point(right_hand_landmarks[mp_holistic.HandLandmark.INDEX_FINGER_MCP], img.shape)
                    wrist_width = int(np.linalg.norm(np.array(pinky_coord) - np.array(index_coord)))

                # Draw arms if landmarks are available
                if landmarks is not None:  # Ensure landmarks are defined
                    for side in ['RIGHT', 'LEFT']:
                        shoulder_pt = get_point(landmarks[getattr(mp_holistic.PoseLandmark, f'{side}_SHOULDER')], img.shape)
                        elbow_pt = get_point(landmarks[getattr(mp_holistic.PoseLandmark, f'{side}_ELBOW')], img.shape)
                        wrist_pt = get_point(landmarks[getattr(mp_holistic.PoseLandmark, f'{side}_WRIST')], img.shape)

                        arm_points = np.array([shoulder_pt, elbow_pt, wrist_pt], np.int32).reshape((-1, 1, 2))
                        overlay = img.copy()
                        cv2.polylines(overlay, [arm_points], isClosed=False, color=(0, 255, 0), thickness=wrist_width)
                        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

                right_shoulder_pt = get_point(landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER], img.shape)
                right_elbow_pt = get_point(landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW], img.shape)
                right_wrist_pt = get_point(landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST], img.shape)

                right_arm_points = np.array([right_shoulder_pt, right_elbow_pt, right_wrist_pt], np.int32)
                right_arm_points = right_arm_points.reshape((-1, 1, 2))

                # Define and draw the left arm (shoulder, elbow, wrist)
                left_shoulder_pt = get_point(landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER], img.shape)
                left_elbow_pt = get_point(landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW], img.shape)
                left_wrist_pt = get_point(landmarks[mp_holistic.PoseLandmark.LEFT_WRIST], img.shape)

                left_arm_points = np.array([left_shoulder_pt, left_elbow_pt, left_wrist_pt], np.int32)
                left_arm_points = left_arm_points.reshape((-1, 1, 2))
                # Create an overlay and add translucent polygons for both arms and chest area
                overlay = img.copy()

                # Draw arms
                cv2.polylines(overlay, [right_arm_points], isClosed=False, color=(0, 255, 0), thickness=10)
                cv2.polylines(overlay, [left_arm_points], isClosed=False, color=(0, 255, 0), thickness=10)

                if not UP and points[24][1] > points[26][1]:
                    pos = "DOWN"
                    print(pos)
                    UP = True
                    target_position_ratio = 0.0  # DOWN position
                    arrow_direction = "UP"
                    show_arrows = True
                elif UP and points[24][1] + 50 < points[26][1]:
                    pos = "UP"
                    print(pos)
                    UP = False
                    counter += 1
                    target_position_ratio = 1.0  # UP position
                    arrow_direction = "UP"
                    show_arrows = True
                    print(counter, "\n------------------\n")
                elif points[15][0] < points[16][0]:
                    print("Exit\nThanks for using this program")
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                elif points[11][0] < points[16][0]:
                    goto()
                if show_arrows:
                    for pt in [24, 23]:
                        if arrow_direction == "UP":
                            cv2.arrowedLine(img, (points[pt][0], points[pt][1] - 50), points[pt], (0, 0, 255), 3,
                                            cv2.LINE_AA, tipLength=0.4)
                        elif arrow_direction == "DOWN":
                            cv2.arrowedLine(img, (points[pt][0], points[pt][1] + 50), points[pt], (0, 0, 255), 3,
                                            cv2.LINE_AA, tipLength=0.4)
                # Smoothly update current_position_ratio toward target_position_ratio
                if current_position_ratio < target_position_ratio:
                    current_position_ratio = min(current_position_ratio + step_size, target_position_ratio)
                elif current_position_ratio > target_position_ratio:
                    current_position_ratio = max(current_position_ratio - step_size, target_position_ratio)
                draw_tracker(img, current_position_ratio)
                # Reset `show_arrows` after the transition to make arrows persist
                if current_position_ratio == target_position_ratio:
                    show_arrows = False
            cv2.putText(img, str(counter) + "  " + pos, (100, 150), cv2.FONT_HERSHEY_PLAIN, 6, (0, 0, 255))
            cv2.imshow('img', img)
            cv2.waitKey(1)
def Press():
    cap = cv2.VideoCapture(0)
    pos = ""
    UP = False
    counter = 0
    current_position_ratio = 0.0
    step_size = 0.05  # Controls the speed of the transition
    target_position_ratio = 0.0
    show_arrows = False
    arrow_direction = "UP"
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            success, img = cap.read()
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = holistic.process(imgRGB)
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
                points = {}
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # print(id,lm,cx,cy)
                    points[id] = (cx, cy)

                cv2.circle(img, points[12], 15, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, points[16], 15, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, points[11], 15, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, points[15], 15, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, points[16], 15, (255, 0, 0), cv2.FILLED)
                wrist_width = 10  # Default thickness in case landmarks are missing
                if results.right_hand_landmarks:
                    right_hand_landmarks = results.right_hand_landmarks.landmark
                    wrist_coord = get_point(right_hand_landmarks[mp_holistic.HandLandmark.WRIST], img.shape)
                    pinky_coord = get_point(right_hand_landmarks[mp_holistic.HandLandmark.PINKY_MCP], img.shape)
                    index_coord = get_point(right_hand_landmarks[mp_holistic.HandLandmark.INDEX_FINGER_MCP], img.shape)
                    wrist_width = int(np.linalg.norm(np.array(pinky_coord) - np.array(index_coord)))

                # Draw arms if landmarks are available
                if landmarks is not None:  # Ensure landmarks are defined
                    for side in ['RIGHT', 'LEFT']:
                        shoulder_pt = get_point(landmarks[getattr(mp_holistic.PoseLandmark, f'{side}_SHOULDER')], img.shape)
                        elbow_pt = get_point(landmarks[getattr(mp_holistic.PoseLandmark, f'{side}_ELBOW')], img.shape)
                        wrist_pt = get_point(landmarks[getattr(mp_holistic.PoseLandmark, f'{side}_WRIST')], img.shape)

                        arm_points = np.array([shoulder_pt, elbow_pt, wrist_pt], np.int32).reshape((-1, 1, 2))
                        overlay = img.copy()
                        cv2.polylines(overlay, [arm_points], isClosed=False, color=(0, 255, 0), thickness=wrist_width)
                        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

                right_shoulder_pt = get_point(landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER], img.shape)
                right_elbow_pt = get_point(landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW], img.shape)
                right_wrist_pt = get_point(landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST], img.shape)

                right_arm_points = np.array([right_shoulder_pt, right_elbow_pt, right_wrist_pt], np.int32)
                right_arm_points = right_arm_points.reshape((-1, 1, 2))

                # Define and draw the left arm (shoulder, elbow, wrist)
                left_shoulder_pt = get_point(landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER], img.shape)
                left_elbow_pt = get_point(landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW], img.shape)
                left_wrist_pt = get_point(landmarks[mp_holistic.PoseLandmark.LEFT_WRIST], img.shape)

                left_arm_points = np.array([left_shoulder_pt, left_elbow_pt, left_wrist_pt], np.int32)
                left_arm_points = left_arm_points.reshape((-1, 1, 2))

                # Define chest area points (shoulders and hips)
                right_hip_pt = get_point(landmarks[mp_holistic.PoseLandmark.RIGHT_HIP], img.shape)
                left_hip_pt = get_point(landmarks[mp_holistic.PoseLandmark.LEFT_HIP], img.shape)

                # Define chest points using an array of tuples for fillPoly
                chest_points = np.array([
                    left_shoulder_pt,
                    right_shoulder_pt,
                    [(right_shoulder_pt[0]+right_hip_pt[0])/2,(right_shoulder_pt[1]+right_hip_pt[1])/2],
                    [(left_shoulder_pt[0]+left_hip_pt[0])/2,(left_shoulder_pt[1]+left_hip_pt[1])/2]
                ], np.int32)

                # Create an overlay and add translucent polygons for both arms and chest area
                overlay = img.copy()

                # Draw arms
                cv2.polylines(overlay, [right_arm_points], isClosed=False, color=(0, 255, 0), thickness=10)
                cv2.polylines(overlay, [left_arm_points], isClosed=False, color=(0, 255, 0), thickness=10)

                # Draw chest area
                cv2.fillPoly(overlay, [chest_points], color=(0, 0, 255))  # Fill the chest area with red
                cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

                if not UP and (points[12][1] - 50 < points[16][1] and points[11][1] - 50 < points[15][1]):
                    pos = "DOWN"
                    print(pos)
                    UP = True
                    counter += 1
                    target_position_ratio = 1.0  # DOWN position
                    arrow_direction = "DOWN"
                    show_arrows = True
                    print(counter, "\n------------------\n")
                elif UP and (points[12][1] - 200 > points[16][1] and points[11][1] - 200 > points[15][1]):
                    pos = "UP"
                    print(pos)
                    target_position_ratio = 0.0  # UP position
                    arrow_direction = "UP"
                    show_arrows = True
                    UP = False
                elif points[15][0] < points[16][0]:
                    print("Exit\nThanks for using this program")
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                elif points[11][0] < points[16][0]:
                    goto()
                if show_arrows:
                    for pt in [16, 15]:
                        if arrow_direction == "UP":
                            cv2.arrowedLine(img, (points[pt][0], points[pt][1] - 50), points[pt], (0, 0, 255), 3,
                                            cv2.LINE_AA, tipLength=0.4)
                        elif arrow_direction == "DOWN":
                            cv2.arrowedLine(img, (points[pt][0], points[pt][1] + 50), points[pt], (0, 0, 255), 3,
                                            cv2.LINE_AA, tipLength=0.4)
                # Smoothly update current_position_ratio toward target_position_ratio
                if current_position_ratio < target_position_ratio:
                    current_position_ratio = min(current_position_ratio + step_size, target_position_ratio)
                elif current_position_ratio > target_position_ratio:
                    current_position_ratio = max(current_position_ratio - step_size, target_position_ratio)
                draw_tracker(img, current_position_ratio)
                # Reset `show_arrows` after the transition to make arrows persist
                if current_position_ratio == target_position_ratio:
                    show_arrows = False
            cv2.putText(img, str(counter) + "  " + pos, (100, 150), cv2.FONT_HERSHEY_PLAIN, 6, (0, 0, 255))
            cv2.imshow('img', img)
            cv2.waitKey(1)
def deadlift():# check karna
    cap = cv2.VideoCapture(0)
    pos = ""
    UP = False
    counter = 0
    current_position_ratio = 0.0
    step_size = 0.05  # Controls the speed of the transition
    target_position_ratio = 0.0
    show_arrows = False
    arrow_direction = "UP"
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            success, img = cap.read()
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = holistic.process(imgRGB)
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
                points = {}
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # print(id,lm,cx,cy)
                    points[id] = (cx, cy)

                cv2.circle(img, points[23], 15, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, points[25], 15, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, points[24], 15, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, points[26], 15, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, points[15], 15, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, points[16], 15, (255, 0, 0), cv2.FILLED)
                cv2.line(img, points[12], points[24] ,(0, 255, 0), 10)
                cv2.line(img, points[11], points[23] ,(0, 255, 0), 10)
                cv2.line(img, points[24], points[26] ,(0, 255, 0), 50)
                cv2.line(img, points[23], points[25] ,(0, 255, 0), 50)
                wrist_width = 10  # Default thickness in case landmarks are missing
                if results.right_hand_landmarks:
                    right_hand_landmarks = results.right_hand_landmarks.landmark
                    wrist_coord = get_point(right_hand_landmarks[mp_holistic.HandLandmark.WRIST], img.shape)
                    pinky_coord = get_point(right_hand_landmarks[mp_holistic.HandLandmark.PINKY_MCP], img.shape)
                    index_coord = get_point(right_hand_landmarks[mp_holistic.HandLandmark.INDEX_FINGER_MCP], img.shape)
                    wrist_width = int(np.linalg.norm(np.array(pinky_coord) - np.array(index_coord)))

                # Draw arms if landmarks are available
                if landmarks is not None:  # Ensure landmarks are defined
                    for side in ['RIGHT', 'LEFT']:
                        shoulder_pt = get_point(landmarks[getattr(mp_holistic.PoseLandmark, f'{side}_SHOULDER')], img.shape)
                        elbow_pt = get_point(landmarks[getattr(mp_holistic.PoseLandmark, f'{side}_ELBOW')], img.shape)
                        wrist_pt = get_point(landmarks[getattr(mp_holistic.PoseLandmark, f'{side}_WRIST')], img.shape)

                        arm_points = np.array([shoulder_pt, elbow_pt, wrist_pt], np.int32).reshape((-1, 1, 2))
                        overlay = img.copy()
                        cv2.polylines(overlay, [arm_points], isClosed=False, color=(0, 255, 0), thickness=wrist_width)
                        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

                right_shoulder_pt = get_point(landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER], img.shape)
                right_elbow_pt = get_point(landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW], img.shape)
                right_wrist_pt = get_point(landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST], img.shape)

                right_arm_points = np.array([right_shoulder_pt, right_elbow_pt, right_wrist_pt], np.int32)
                right_arm_points = right_arm_points.reshape((-1, 1, 2))

                # Define and draw the left arm (shoulder, elbow, wrist)
                left_shoulder_pt = get_point(landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER], img.shape)
                left_elbow_pt = get_point(landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW], img.shape)
                left_wrist_pt = get_point(landmarks[mp_holistic.PoseLandmark.LEFT_WRIST], img.shape)

                left_arm_points = np.array([left_shoulder_pt, left_elbow_pt, left_wrist_pt], np.int32)
                left_arm_points = left_arm_points.reshape((-1, 1, 2))
                # Create an overlay and add translucent polygons for both arms and chest area
                overlay = img.copy()

                # Draw arms
                cv2.polylines(overlay, [right_arm_points], isClosed=False, color=(0, 255, 0), thickness=10)
                cv2.polylines(overlay, [left_arm_points], isClosed=False, color=(0, 255, 0), thickness=10)

                if not UP and points[25][1] > points[15][1]:
                    pos = "UP"
                    print(pos)
                    UP = True
                    target_position_ratio = 0.0  # UP position
                    arrow_direction = "UP"
                    show_arrows = True
                elif UP and points[25][1] + 70  < points[15][1]:
                    pos = "DOWN"
                    print(pos)
                    UP = False
                    counter += 1
                    target_position_ratio = 1.0  # DOWN position
                    arrow_direction = "DOWN"
                    show_arrows = True
                    print(counter, "\n------------------\n")
                elif points[15][0] < points[16][0]:
                    print("Exit\nThanks for using this program")
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                elif points[11][0] < points[16][0]:
                    goto()
                if show_arrows:
                    for pt in [16, 15]:
                        if arrow_direction == "UP":
                            cv2.arrowedLine(img, (points[pt][0], points[pt][1] - 50), points[pt], (0, 0, 255), 3,
                                            cv2.LINE_AA, tipLength=0.4)
                        elif arrow_direction == "DOWN":
                            cv2.arrowedLine(img, (points[pt][0], points[pt][1] + 50), points[pt], (0, 0, 255), 3,
                                            cv2.LINE_AA, tipLength=0.4)
                # Smoothly update current_position_ratio toward target_position_ratio
                if current_position_ratio < target_position_ratio:
                    current_position_ratio = min(current_position_ratio + step_size, target_position_ratio)
                elif current_position_ratio > target_position_ratio:
                    current_position_ratio = max(current_position_ratio - step_size, target_position_ratio)
                draw_tracker(img, current_position_ratio)
                # Reset `show_arrows` after the transition to make arrows persist
                if current_position_ratio == target_position_ratio:
                    show_arrows = False
            cv2.putText(img, str(counter) + "  " + pos, (100, 150), cv2.FONT_HERSHEY_PLAIN, 6, (0, 0, 255))
            cv2.imshow('img', img)
            cv2.waitKey(1)

def goto():
    exercise_input = int(input("Instructions -:\n-Place you camera directly in front of you."
                               "\n-If you want to change your Exercise , tap on you left shoulder "
                               "with your right hand.\n"
                               "with your left hand.\n-If you want to quit ,"
                               " make a cross with your arms.\n\n\n"
                               "Choose your your type of Exerice-\n"
                               "1) Weighted exercise \n2) Non Weighted exercise"
                               "\nEnter a valid number from above"))
    if exercise_input== 2 :
        nwex= int(input("Enter your Weighted exercise:\n1)Push Up \n2)Pull up \n3)Squats"
                     "\nEnter a valid number from above"))
        if nwex == 1:#Push UP
            PushUp()
        elif nwex == 2:#pull up
            PullUp()
        elif nwex == 3:#squats
            Squats()
        else:
            print("wrong input")
            ans = int(input("If you want to quit press 1\nIf you want to continue press 0"))
            if ans == 1:
                print("Thanks for using this program")
                cap.release()
                cv2.destroyAllWindows()
                return
            elif ans == 0:
                goto()
    if exercise_input==1:
        wex= int(input("Enter your Non Weighted exercise:\n1)Bicep Curls"
                     "\n2)Shoulder raises(any type)"
                     "\n3)Shoulder/inclined Press \n4)Deadlift"
                     "\nEnter a valid number from above"))
        if wex == 1:  # curls
            Curls()
        elif wex == 2:  # shoulder raises
            Shoulder_raises()
        elif wex == 3:  # press
            Press()
        elif wex == 4: #deadlift
            deadlift()
        else:
            print("wrong input")
            ans = int(input("If you want to quit press 1\nIf you want to continue press 0"))
            if ans == 1:
                print("Thanks for using this program")
                cap.release()
                cv2.destroyAllWindows()
                return
            elif ans == 0:
                goto()
    else:
        print("wrong input")
        ans = int(input("If you want to quit press 1\nIf you want to continue press 0"))
        if ans == 1:
            print("Thanks for using this program")
            cap.release()
            cv2.destroyAllWindows()
            return
        elif ans == 0:
            goto()
goto()
tf.get_logger().setLevel('ERROR')  # Set logging level to ERROR to avoid warnings
