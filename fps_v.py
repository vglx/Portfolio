import time
import torch
import cv2

# 加载模型
model = torch.hub.load('./', 'custom', path='origin', source='local')  # 加载自定义模型

video = cv2.VideoCapture('vid1.mp4')

num_frames = 100  # 测量FPS的帧数
start_time = time.time()

for i in range(num_frames):
    ret, frame = video.read()
    if not ret:
        break

    # 使用YOLOv5进行对象检测
    results = model(frame)

end_time = time.time()

# 计算并打印FPS
fps = num_frames / (end_time - start_time)
print(f'FPS: {fps}')
print(f'VFPS: {video.get(cv2.CAP_PROP_FPS)}')

# 释放视频和销毁所有窗口
video.release()
cv2.destroyAllWindows()