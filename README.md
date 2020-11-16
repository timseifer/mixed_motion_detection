[cars-yolo-output]: examples/assets/cars.gif "Sample Output with YOLO"
[cows-tf-ssd-output]: examples/assets/cows.gif "Sample Output with SSD"

# Multi-object trackers in Python
Object detection using deep learning and multi-object tracking

[![DOI](https://zenodo.org/badge/148338463.svg)](https://zenodo.org/badge/latestdoi/148338463)

### Available Multi Object Trackers

```
CentroidTracker
CentroidKF_Tracker
IOUTracker
SORT
```

### Available OpenCV-based object detectors:

```
detector.TF_SSDMobileNetV2
detector.Caffe_SSDMobileNet
detector.YOLOv3
```

`YOLOv3 + CentroidTracker` |  `TF-MobileNetSSD + CentroidTracker`
:-------------------------:|:-------------------------:
![Cars with YOLO][cars-yolo-output]  |  ![Cows with tf-SSD][cows-tf-ssd-output]
Video source: [link](https://flic.kr/p/L6qyxj) | Video source: [link](https://flic.kr/p/26WeEWy)


### Installation
Pip install for OpenCV (version 3.4.3 or later) is available [here](https://pypi.org/project/opencv-python/) and can be done with the following command:

```
pip install numpy matplotlib scipy
pip install opencv-contrib-python
```

Installation of `ipyfilechooser` is recommended if you want to use the jupyter notebooks available in the ```examples``` folder.
```
pip install ipyfilechooser
```

```
git clone https://github.com/adipandas/multi-object-tracker
cd multi-object-tracker
pip install -e .
```

For using the opencv `dnn`-based object detection modules provided in this repository with GPU, you may have to compile a CUDA enabled version of OpenCV from source.

For building opencv from source, you can refer the following: [[link-1](https://docs.opencv.org/master/df/d65/tutorial_table_of_content_introduction.html)], [[link-2](https://www.pyimagesearch.com/2020/02/03/how-to-use-opencvs-dnn-module-with-nvidia-gpus-cuda-and-cudnn/)]

### How to use?: Examples

Examples for how to use `motrackers` are provided [examples](./examples/) folder of this repository.
You can clone and run the examples as shown in the [readme](examples/readme.md) inside the [examples](./examples/) folder.

### Pretrained Object Detection Models

You will have to download the pretrained weights for the model. The shell scripts for downloading are provided in [examples](examples/) folder.

##### YOLOv3
```
cd ./examples/pretrained_models/yolo_weights
sudo chmod +x ./get_yolo.sh
./get_yolo.sh
```

##### TensorFlow MobileNetSSDv2
```
cd ./pretrained_models/tensorflow_weights
sudo chmod +x ./get_ssd_model.sh
./get_ssd_model.sh
```

##### Caffemodel
```
cd ./pretrained_models/caffemodel_weights
sudo chmod +x ./get_caffemodel.sh
./get_caffemodel.sh
```

### References and Credits

Please see [references.md](REFERENCES.md)

### Citation

If you use this repository in your work, please consider citing it with:
```
@misc{multiobjtracker_amd2018,
  author = {Deshpande, Aditya M.},
  title = {Multi-object trackers in Python},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/adipandas/multi-object-tracker}},
}
```

```
@software{aditya_m_deshpande_2020_3951169,
  author       = {Aditya M. Deshpande},
  title        = {Multi-object trackers in Python},
  month        = jul,
  year         = 2020,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.3951169},
  url          = {https://doi.org/10.5281/zenodo.3951169}
}
```
