import torch
import cv2

model = torch.hub.load("./", "custom", path='originalYOLOv5', source="local")

video = cv2.VideoCapture('a.mp4')

# 迭代处理每一帧
while True:
    ret, frame = video.read()
    if not ret:
        break

    # 使用 YOLOv5 进行对象检测
    results = model(frame)

    # 获取检测到的对象和它们的位置
    for *xyxy, conf, cls in results.xyxy[0]:
        # 在图像上绘制检测到的对象
        cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)

    # 显示图像
    cv2.imshow('frame', frame)

    # 等待 1 毫秒
    if cv2.waitKey(1) == ord('q'):
        break

# 释放视频和销毁所有窗口
video.release()
cv2.destroyAllWindows()