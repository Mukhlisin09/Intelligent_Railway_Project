**A YOLOv8n-Based Method for an Intelligent Railway Level Crossing System with Real-Time Vehicle Detection**

This repository contains the source code and implementation of a YOLOv8n-based real-time vehicle detection method developed for an intelligent railway level crossing system. The method is designed to detect vehicles within the railway crossing area using computer vision and provide real-time information to support automated level crossing safety mechanisms.

The implementation uses YOLOv8n (YOLOv8 Nano) to achieve a balance between detection accuracy and computational efficiency, making the method suitable for real-time and edge-computing applications.

**Method Overview**
The proposed method consists of the following main stages:
1. Image acquisition from a camera installed at the railway level crossing.
2. Image preprocessing and preparation of input frames.
3. Vehicle detection using YOLOv8n.
4. Region of Interest (ROI) analysis to determine whether detected vehicles are located within the railway crossing area.
5. Real-time detection and monitoring of vehicles in the crossing area.
6. Occupancy assessment of the railway crossing area based on the detection results.
7. Safety decision support for an intelligent railway level crossing system.
8. Supported Object Detection

The model can be trained and configured to detect relevant objects in the railway crossing environment, such as:
a. Cars
b. Motorcycles
c. Buses
d. Trucks
e. Other road vehicles
The object classes can be modified according to the dataset and application requirements.

**Repository Contents**
The repository provides the implementation required to reproduce the vehicle detection method, including:
YOLOv8n model configuration
Training and validation scripts
Inference and real-time detection scripts
Dataset preparation utilities
ROI configuration
Detection result processing
Performance evaluation
Example configuration files
Research Purpose

This repository is provided to support the reproducibility of the methodology presented in the associated MethodsX article:
"A YOLOv8n-Based Method for an Intelligent Railway Level Crossing System with Real-Time Vehicle Detection"

**Citation
**
If you use this code or methodology in your research, please cite the associated MethodsX article:
A YOLOv8n-Based Method for an Intelligent Railway Level Crossing System with Real-Time Vehicle Detection.
