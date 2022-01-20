
# img  = cv2.imread('lena.png')
#
#
# className = []
# classFile = 'coco.names'
# with open(classFile , 'rt') as f:
#     className = f.read().rstrip('\n').split('\n')
#
# cfgpath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
# weightPath = 'frozen_inference_graph_1.pb'
#
# net = cv2.dnn_DetectionModel(weightPath, cfgpath)
# net.setInputSize(320,320)
# net.setInputScale(1.0 / 127.5)
# net.setInputMean((127.5,127.5,127.5))
# net.setInputSwapRB(True)
#
# classIds , confs , bbox = net.detect(img , confThreshold= 0.5)
#
# print(classIds,bbox)
#
# cv2.imshow("output" , img)
# cv2.waitKey(0)


import cv2
import matplotlib.pyplot as plt


cap = cv2.VideoCapture()
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

img = cv2.imread('lena.png')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

ClassIndex, confidence, bbox = model.detect(img, confThreshold=0.5)

for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
    cv2.rectangle(img, boxes, (255, 0, 0), thickness=1)
    cv2.putText(img, classlabels[ClassInd-1].upper(), (boxes[0]+10, boxes[1]+40),
                cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0, 0, 0 ), thickness=2)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
cv2.imshow("output" , img)
cv2.waitKey(0)
