# Coordinate_HS_YOLOv5s

This is a modified deep learning network modified on the basis of YOLOv5 object detection network.
<br/>

The network implementation refers to [YOLOv5](https://github.com/ultralytics/yolov5).
<br/>

The following figure shows the architecture of our network. The flowchart design refers to [this Chinese blog](https://blog.csdn.net/nan355655600/article/details/107852353).
![image](https://user-images.githubusercontent.com/40823626/185787909-07055f57-c704-4142-b0cd-fa2909e64581.png)
<br/>

The two important modules introduced are [CoordConv module](https://arxiv.org/pdf/1807.03247.pdf) and [Hierarchical-Split block module](https://arxiv.org/pdf/2010.07621.pdf) (HS block), respectively. the CoordConv module adds a channel representing the coordinate position to the input image, and the HS block perform multi-scale separation and convolution on the feature map to deepen the number of feature layers and reuse the feature map.
<br/>

## impletation details

Since the data sets we conduct model training and model verification are widely used open-source driving datasets, this project does not include these data sets. When using our network, **pay attention to modifying the yaml file** (addr: **./data/dataset_info.yaml**) parameters (including strings of dataset address, integer of number of classes, and list of class's name) that control the reading of datasets.
<br/>


**hyper-parameters**

| Hyper-parameters          | Value                                           |
| ------------------------- | ----------------------------------------------- |
| training epoch            | 150                                             |
| learning rate             | 0.01                                            |
| batch size                | 64                                              |
| optimizer                 | SGD (momentum at .937 and weight decay at 4e-5) |
| NMS IoU overlap threshold | 0.45                                            |
<br/>


The training saved model weights are saved in **./weights/coord_hs_yolov5s.pt**
<br/>


## requirements
```
Cython
matplotlib>=3.2.2
numpy>=1.18.5
opencv-python>=4.1.2
Pillow
PyYAML>=5.3
scipy>=1.4.1
tensorboard>=2.2
torch>=1.7.0
torchvision>=0.8.1
tqdm>=4.41.0
seaborn>=0.11.0
pandas
thop
pycocotools>=2.0
```
<br/>


## network performance

The network has learned on many driving data sets, and the learning results are as follows:

| Dataset    | APs     | YOLOv3 tiny | YOLOv5s  | Coord YOLOv5s | HS-YOLOv5s | Coord HS-YOLOv5s |
| ---------- | ------- | ----------- | -------- | ------------- | ---------- | ---------------- |
| BDD100K    | overall | 68.2        | 83.0     | 83.2          | 83.5       | **83.6**         |
|            | person  | 55.4        | 75.6     | 75.7          | **76.2**   | 76.1             |
|            | car     | 81.0        | 90.7     | 90.8          | 90.8       | **91.1**         |
| KITTI      | overall | 74.3        | 87.5     | 88.0          | 88.1       | **88.3**         |
|            | person  | 60.8        | 78.9     | 80.4          | 80.7       | **80.8**         |
|            | car     | 87.8        | **96.1** | 95.7          | 95.5       | 95.8             |
| CityScapes | overall | 60.6        | 72.0     | 72.3          | 72.7       | **72.8**         |
|            | person  | 50.7        | 64.5     | 65.1          | **65.8**   | 65.5             |
|            | car     | 70.5        | 79.4     | 79.5          | 79.6       | **80.0**         |
<br/>

while the detection speed of each models are,

| Networks         | Speed (ms per image) |
| ---------------- | -------------------- |
| YOLOv3 tiny      | **6.2**              |
| YOLOv5s          | 9.7                  |
| Coord YOLOv5     | 11.2                 |
| HS-YOLOv5s       | 13.3                 |
| Coord HS-YOLOv5s | 13.8                 |
<br/>


In addition, in order to verify the detection effect of the network under special weather images, we also extracted the rain and snow weather images in BDD100K and used them as a data set for network learning and verification. The image extraction method is to recognize the images with weather categories marked as 'rainy' or 'snowy' in BDD100K. The performances of each network are shown as follows:


| Networks         | AP-overall | AP-person | AP-car   |
| ---------------- | ---------- | --------- | -------- |
| YOLOv3 tiny      | 50.1       | 41.3      | 58.9     |
| YOLOv5s          | 67.4       | 55.6      | 79.1     |
| Coord YOLOv5     | 68.4       | 57.2      | **79.7** |
| HS-YOLOv5s       | 68.4       | 57.3      | 79.5     |
| Coord HS-YOLOv5s | **68.6**   | **57.5**  | **79.7** |
<br/>


For any problems, please contact us by zhangyingjie2020@email.szu.edu.cn or cyychenyaoyu@163.com
