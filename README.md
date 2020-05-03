# Final Project of GPU Parallel Processing
## MTCNN Faces/Heads Detection in image/video
## Intruduction: see FinalProject slides.
==================================

## Set up

Set up environment and copy C++ layer code to Caffe's source code tree.

```
$ export PYTHONPATH=/path/to/mtcnn:$PYTHONPATH
$ export CAFFE_HOME=/path/to/caffe
$ pip install easydict
$ pip install lmdb
$ sh layers/copy.sh
```

Compile Caffe following its document.

## Prepare data

Download dataset [SCUT-HEAD](https://github.com/HCIILAB/SCUT-HEAD-Dataset-Release).
Unzip and put them in data directory.

Add faces with mask dataset by yourself. Find and label faces with mask pictures from google or anywhere.

## Train

**pnet**
```
python jfda/prepare.py --net p --wider --worker 8
python jfda/train.py --net p --gpu 0 --size 128 --lr 0.05 --lrw 0.1 --lrp 5 --wd 0.0001 --epoch 25
```
**rnet**

Choose appropriate pnet caffemodel to generate prior for rnet, and edit ```cfg.PROPOSAL_NETS``` in ```config.py```
```
python jfda/prepare.py --net r --gpu 0 --detect --wider --worker 4
python jfda/train.py --net r --gpu 0 --size 128 --lr 0.05 --lrw 0.1 --lrp 5 --wd 0.0001 --epoch 25
```
**onet**

Choose appropriate rnet caffemodel to generate prior for onet, and edit ```cfg.PROPOSAL_NETS``` in ```config.py```
```
python jfda/prepare.py --net o --gpu 0 --detect --wider --worker 4
python jfda/train.py --net o --gpu $GPU --size 64 --lr 0.05 --lrw 0.1 --lrp 7 --wd 0.0001 --epoch 35
```

## Test

```
python simpledemo.py
```

## Note

1. Landmark alignment in original mtcnn is removed in this repo. Here only do object classification and bounding box regression. 

2. Each convolutional layer kernel number in onet has reduced for faster network inference.

## Results

**pnet**

![pnet1](https://github.com/zren2/GPU-PARALLEL-PROCESSING/blob/master/FinalProject/2020-05-01%2014-19-56%20%E7%9A%84%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png)

**rnet**

![rnet1](https://github.com/zren2/GPU-PARALLEL-PROCESSING/blob/master/FinalProject/2020-05-01%2014-20-07%20%E7%9A%84%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png)

**onet**

![onet1](https://github.com/zren2/GPU-PARALLEL-PROCESSING/blob/master/FinalProject/2020-05-01%2014-20-22%20%E7%9A%84%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png)

**Detect Faces With Mask**

![onet1](https://github.com/zren2/GPU-PARALLEL-PROCESSING/blob/master/FinalProject/2020-04-29%2011-19-27%20%E7%9A%84%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png)
![onet1](https://github.com/zren2/GPU-PARALLEL-PROCESSING/blob/master/FinalProject/2020-04-29%2011-20-17%20%E7%9A%84%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png)

**Detect heads**

![onet1](https://user-images.githubusercontent.com/16308037/53081747-6c1f2f80-3536-11e9-84bc-6885cf991468.jpg)
