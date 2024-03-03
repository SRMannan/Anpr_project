import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO



model = YOLO('../yolov8s.pt')


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)


#cv2.namedWindow('RGB')
#cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('vidyolov8.mp4')

my_file = open("../coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
print(class_list)
count = 0

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1020, 500))
    if ret is None:
        break
    count += 1
    if count % 3 != 0:
        continue

    xyxy = []
    conf = []
    cls = []

    results = model.predict(frame)
    ##extracting BB coordinates
    for r in results:
        boxes = r.boxes.cpu().numpy()
        xyxy = boxes.xyxy
        conf = boxes.conf
        cls = boxes.cls
    ##BB coord strored in a DataFrame
    p_cord = pd.DataFrame(xyxy)
    p_conf = pd.DataFrame(conf)
    p_cls = pd.DataFrame(cls)

    ##print("\nBOXES" , xyxy)
    #print("\nCONF" , conf)
    #print("\nCLS" , cls)

    ##Extracting BBs
    for (cord_index, cord_row), (cls_index, cls_row) in zip(p_cord.iterrows(), p_cls.iterrows()):
        x1 = int(cord_row[0])
        y1 = int(cord_row[1])
        x2 = int(cord_row[2])
        y2 = int(cord_row[3])
        obj_cls = int(cls_row[0])
        c = class_list[obj_cls]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, str(c), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
