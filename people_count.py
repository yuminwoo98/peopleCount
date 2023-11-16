import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import math
import time
from sort import *    #sort 파일은 하단에 토글로 따로 참고.

cap = cv2.VideoCapture("../Videos/people.mp4")  # For Video
# 동영상 파일 읽기

model = YOLO(
    "C:\JIN\opencv_study\object-Detection\object-Detection\Yolo-Weights\yolov8n.pt"
)
# YOLO 모델을 초기화하고, 해당 모델을 사용하여 입력 이미지 
#또는 비디오에서 객체를 검출하고 분류하기 위함.



classNames = [
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "diningtable",
    "toilet",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]
mask = cv2.imread("mask.png")   
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
tracker2 = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limits = [400, 297, 673, 297]       # 라인1
limits2 = [140, 297, 399, 297]      # 라인2
totalCount = []  # 내려간다
totalCount2 = []   # 올라간다

while True:
    success, img = cap.read()    
		# cap 동영상을 읽어온다. 프레임을 성공적으로 읽어오면 success는 True 가 된다
		# img : 읽어온 프레임 데이터가 이 변수에 저장된다.
    imgRegion = cv2.bitwise_and(img, mask)
		# cv2.bitwise_and 함수는 이미지 간의 비트 단위 AND 연산을 수행한다.
		# 읽어온 프레임 mask와 img를 사용해 지정된 영역 내에서 객체검출을 함.
    results = model(imgRegion, stream=True)
    detections = np.empty((0, 5))

    imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (750,300))  # 그래픽png 파일 위치 
    if not success:
        break
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            # print("tensor x1, y1, x2, y2", x1, y1, x2, y2)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            conf = math.ceil((box.conf[0] * 100)) / 100

            if (
                currentClass == "person"

                and conf > 0.3
            ):
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
                resultsTracker = tracker.update(detections)
                resultsTracker2 = tracker2.update(detections)

                cv2.line(
                    img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5  
                     #        내려가는 선 그리기       
                )
                cv2.line(
                    img, (limits2[0], limits2[1]), (limits2[2], limits2[3]), (0, 150, 100), 5  
                     #  올라가는 선 그리기  
                )
								

                for result in resultsTracker:
                    # x1, y1, x2, y2, id,id2 = result
                    x1, y1, x2, y2, id = result
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    print(result)
                    w, h = x2 - x1, y2 - y1
                    cvzone.cornerRect(
                        img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 0)
                    )
                    cvzone.putTextRect(
                        img,
                        f"{int(id)}",
                        (max(0, x1), max(35, y1)),
                        scale=2,
                        thickness=3,
                        offset=10,
                    )
                    cvzone.putTextRect(
                        img,
                        f"{int(id)}",
                        (max(0, x1), max(35, y1)),
                        scale=2,
                        thickness=3,
                        offset=10,
                    )
                    cx, cy = x1 + w // 2, y1 + h // 2
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                    if (
                        limits[0] < cx < limits[2]
                        and limits[1] - 15 < cy < limits[1] + 15
                    ):
                        if totalCount.count(id) == 0:
                            totalCount.append(id)
                            cv2.line(
                                img,
                                (limits[0], limits[1]),
                                (limits[2], limits[3]),
                                (0, 255, 0),
                                5,
                            )
                    if (
                        limits[0] < cx < limits[2]
                        and limits[1] - 15 < cy < limits[1] + 15
                    ):
                        if totalCount.count(id) == 0:
                            totalCount.append(id)
                            cv2.line(
                                img,
                                (limits[0], limits[1]),
                                (limits[2], limits[3]),
                                (0, 255, 0),
                                5,
                            )
                

                #id2 를 사용하기 위해 Tracker2 를 만들어주고 for문 하나 더 만든다.
                for result in resultsTracker2:
                    # x1, y1, x2, y2, id,id2 = result
                    x1, y1, x2, y2, id2 = result
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    print(result)
                    w, h = x2 - x1, y2 - y1
                    cvzone.cornerRect(
                        img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 0)
                    )
                    cvzone.putTextRect(
                        img,
                        f"{int(id2)}",
                        (max(0, x1), max(35, y1)),
                        scale=2,
                        thickness=3,
                        offset=10,
                    )

                    cvzone.putTextRect(
                        img,
                        f"{int(id2)}",
                        (max(0, x1), max(35, y1)),
                        scale=2,
                        thickness=3,
                        offset=10,
                    )

                    cx, cy = x1 + w // 2, y1 + h // 2
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                    if (
                        limits[0] < cx < limits[2]
                        and limits[1] - 15 < cy < limits[1] + 15
                    ):
                        if totalCount.count(id2) == 0:
                            totalCount.append(id2)
                            cv2.line(
                                img,
                                (limits[0], limits[1]),
                                (limits[2], limits[3]),
                                (0, 255, 0),
                                5,
                            )


                    if (
                        limits2[0] < cx < limits2[2]
                        and limits2[1] - 15 < cy < limits2[1] + 15
                    ):
                        if totalCount2.count(id2) == 0:
                            totalCount2.append(id2)
                            cv2.line(
                                img,
                                (limits2[0], limits2[1]),
                                (limits2[2], limits2[3]),
                                (0, 255, 0),
                                5,
                            )

								# 내려간느것을 카운트하고 나타낸다.
                cv2.putText(     
                    img,
                    str(len(totalCount)),
                    (1200, 400),
                    cv2.FONT_HERSHEY_PLAIN,
                    5,
                    (50, 50, 255),
                    8,
                )


								# 올라가는것을 카운트 하고 나타낸다.
                cv2.putText(     
                    img,
                    str(len(totalCount2)),
                    (950, 400),
                    cv2.FONT_HERSHEY_PLAIN,
                    5,
                    (50, 50, 255),
                    8,
                )
                

    cv2.imshow("Image", img)
    cv2.waitKey(1)