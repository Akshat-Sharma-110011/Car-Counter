# 🚗 Car Counter using YOLO and SORT

This project implements a **real-time vehicle counting system** using **YOLOv8** for object detection and **SORT (Simple Online and Realtime Tracker)** for object tracking. 🏎️📊

## 📌 Features
- **YOLOv8 Object Detection**: Detects vehicles like cars, trucks, buses, and motorbikes.
- **SORT Tracker**: Assigns unique IDs to detected vehicles and tracks them across frames.
- **Region Masking**: Uses a mask image to focus detection in a specific region.
- **Vehicle Classification**: Assigns different colors for each vehicle type.
- **Line Crossing Detection**: Counts vehicles crossing a predefined detection line.

## 📂 Requirements
Ensure you have the following dependencies installed:
```bash
pip install numpy opencv-python ultralytics cvzone filterpy
```

## 🔧 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Akshat-Sharma-110011/car-counter.git
   cd car-counter
   ```
2. Download the YOLOv8 model weights:
   ```bash
   wget https://github.com/ultralytics/assets/releases/download/v8/yolov8l.pt
   ```
3. Run the program:
   ```bash
   python car_counter.py
   ```

## 🚀 How It Works
- **Loads YOLOv8** to detect vehicles in each frame.
- **Applies a mask** to filter out unwanted regions.
- **Uses SORT tracking** to maintain unique vehicle IDs across frames.
- **Counts vehicles** when they cross the predefined red line.

## 🎥 Video Input
Place your input video (`cars.mp4`) in the project folder or modify the script to use a live camera feed.

## 🛠️ Customization
- Modify `classNames` to track additional objects.
- Adjust `limits = [x1, y1, x2, y2]` to change the detection line.
- Fine-tune `tracker` parameters for better tracking accuracy.

## 📊 Output
- Detected vehicles are displayed with **bounding boxes** and **unique IDs**.
- Total vehicle count is shown on the screen.
- Press `'Q'` to exit the program.

## 📜 License
This project is open-source and available under the **MIT License**.

Happy Coding! 🚀

