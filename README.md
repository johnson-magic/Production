
<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [PRODUCTION](#production)
  - [1. 使用](#1-使用)
    - [1.1 下载](#11-下载)
    - [1.2 指令](#12-指令)
    - [1.3 结果](#13-结果)
- [2. 相关资源](#2-相关资源)

<!-- /code_chunk_output -->

# PRODUCTION
## 1. 使用
### 1.1 下载
```
链接：https://pan.quark.cn/s/74663448c75f
```
### 1.2 指令
```
.\Production.exe .\text_det.onnx .\text_direction_cla.onnx .\text_rec.onnx .\charset.txt '.\2022-12-08 14-54-28_000149.bmp' res.txt vis.jpg
```
### 1.3 结果
```
Point1 691.914,206.116
Point2 282.269,786.145
Distance 710.1
Angle 54.7684
IsRight 1
```
<img src="./imgs/vis.jpg" height=300 width=300>

# 2. 相关资源

* [有向/通用目标检测训练库: ](https://github.com/johnson-magic/PrecisionAngleDetection)