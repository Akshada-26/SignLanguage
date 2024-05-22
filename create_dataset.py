import os
import pickle
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

# Debug: Print all items in DATA_DIR to see what is being processed
print("Items in DATA_DIR:", os.listdir(DATA_DIR))

for dir_ in os.listdir(DATA_DIR):
    full_dir_path = os.path.join(DATA_DIR, dir_)
    print("Processing:", full_dir_path)  # Debug: Print the path being processed

    if os.path.isdir(full_dir_path):  # Check if it's a directory
        for img_path in os.listdir(full_dir_path):
            print("Processing image:", img_path)  # Debug: Print each image being processed

            data_aux = []
            img = cv2.imread(os.path.join(full_dir_path, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    x_ = [landmark.x for landmark in hand_landmarks.landmark]
                    y_ = [landmark.y for landmark in hand_landmarks.landmark]

                    for i in range(len(hand_landmarks.landmark)):
                        data_aux.extend([hand_landmarks.landmark[i].x - min(x_),
                                         hand_landmarks.landmark[i].y - min(y_)])

                data.append(data_aux)
                labels.append(dir_)

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
