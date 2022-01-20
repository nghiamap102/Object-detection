
import cv2
from django.contrib.messages import success

cap = cv2.VideoCapture(1)
cap.set(3,640)
cap.set(4,480)

classlabels = []
file_name = 'coco.names'
with open(file_name, 'rt') as fpt:
    classlabels = fpt.read().rstrip('\n').split('\n')


frozen_model = 'frozen_inference_graph.pb'
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'


model = cv2.dnn_DetectionModel(frozen_model, config_file)

model.setInputSize(320, 320)
model.setInputScale(1.0/127.5) # 255 / 2 = 127.5
model.setInputMean((127.5, 127.5, 127.5)) # mobilenet => [-1, 1]
model.setInputSwapRB(True)

while True:
    success , img = cap.read()
    ClassIndex, confidence, bbox = model.detect(img, confThreshold=0.5)
    print(ClassIndex , bbox)
    for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
        cv2.rectangle(img, boxes, (255, 0, 0), thickness=1)
        cv2.putText(img, classlabels[ClassInd-1].upper(), (boxes[0]+10, boxes[1]+40),
                    cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0, 0, 0 ), thickness=2)

    cv2.imshow("output" , img)
    cv2.waitKey(0)

