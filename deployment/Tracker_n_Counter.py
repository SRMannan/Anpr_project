import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import *

model = YOLO('../yolov8s.pt')


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('veh2.mp4')

my_file = open("../coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
print(class_list)

count = 0

tracker = Tracker()

cy1 = 322
cy2 = 368
offset = 6
v_d = {}            #dict to store downstream vehichle id and coord
v_u = {}            #dict to store upstream vehichle id and coord
cnt = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    list = []

    for index, row in px.iterrows():
        #        print(row)

        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'car' in c:
            list.append([x1, y1, x2, y2])
    bbox_id = tracker.update(list)
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
        if cy1 < (cy+offset) and cy1 > (cy-offset):
            v_d[id] = cy
        if id in v_d:
            if cy2 < (cy+offset) and cy2 > (cy-offset):
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
                if id not in cnt:
                    cnt.append(id)

        if cy2 < (cy+offset) and cy2 > (cy-offset):
            v_u[id] = cy
        if id in v_u:
            if cy1 < (cy+offset) and cy1 > (cy-offset):
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
                if id not in cnt:
                    cnt.append(id)


    ## to Draw the chkpt lines and texting them
    cv2.line(frame,(241,cy1),(823,cy1),(255,255,255),1)
    cv2.putText(frame, ('CHKPT1'), (281, 317), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 255), 1)
    cv2.putText(frame, ('CHKPT2'), (187, 360), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 255), 1)
    cv2.line(frame,(142,cy2),(927,cy2),(255,255,255),1)

    tc = str(len(cnt))
    cv2.putText(frame , ('Tracker_Count : ') + tc , (25,25) , cv2.FONT_HERSHEY_COMPLEX , 0.5 , (125,125,255) , 2)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(50) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()

