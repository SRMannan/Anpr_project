import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
import subprocess
import easyocr
import os
import shutil

count = 0
file_count = 0

model = YOLO(r'C:\Users\shara\OneDrive\Desktop\pythonProject_jan23\Number_plate_Cus_Models\best.pt')

cap = cv2.VideoCapture(r'C:\Users\shara\OneDrive\Desktop\pythonProject_jan23\Sample_Video\Anpr_sample_1.mp4')

output_folder = 'extracted_frames'
os.makedirs(output_folder, exist_ok=True)

while True:
    ret , frame = cap.read()
    if frame is None or frame.size == 0:
        break
    else:
        frame = cv2.resize(frame, (1020, 500))
    if ret is None:
        break
    count += 1
    if count % 3 != 0:
        continue

    xyxy = []

    results = model.predict(frame)
    for r in results:
        boxes = r.boxes.cpu().numpy()
        xyxy = boxes.xyxy
        conf = boxes.conf
        cls = boxes.cls

    p_cord = pd.DataFrame(xyxy)
    p_conf = pd.DataFrame(conf)
    p_cls = pd.DataFrame(cls)

    ##print("\nBOXES" , xyxy)
    # print("\nCONF" , conf)
    # print("\nCLS" , cls)

    ##Extracting BBs
    for (cord_index, cord_row), (cls_index, cls_row) in zip(p_cord.iterrows(), p_cls.iterrows()):
        x1 = int(cord_row[0])
        y1 = int(cord_row[1])
        x2 = int(cord_row[2])
        y2 = int(cord_row[3])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        img = frame[y1:y2+1 , x1:x2+1]
        file_count = file_count+1

        frame_filename = f'frame_{file_count}.jpg'
        frame_filepath = os.path.join(output_folder, frame_filename)
        cv2.imwrite(frame_filepath, img)


    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
#subprocess.run(['python' , 'NP_Detection_OCR.py' , output_folder])
#shutil.rmtree(output_folder)