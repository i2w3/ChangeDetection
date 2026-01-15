import cv2
from ultralytics import YOLO


if __name__ == "__main__":
    # Load a model
    model = YOLO("./res/yolov8s-seg_LoveDA.pt")

    # Perform inference on an image
    results = model("./res/data/UAV2/im1.jpg", )

    # Show the results
    results[0].show()