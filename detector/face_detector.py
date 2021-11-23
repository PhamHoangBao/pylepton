
from tflite_runtime.interpreter import Interpreter
import cv2
import numpy as np

class ThermalFaceDetector():
    def __init__(self, model_path):
        self.detector = Interpreter(model_path)
        self.input_shape = self.detector.get_input_details()[0]["shape"]
        self.output_details = self.detector.get_output_details()
        self.input_details = self.detector.get_input_details()[0]
    def __call__(self, thermal_img):
        thermal_img = cv2.resize(thermal_img, 
                    (self.input_shape[1], self.input_shape[2]), 
                    interpolation = cv2.INTER_NEAREST)
        thermal_img = thermal_img.astype(self.input_details["dtype"])
        thermal_img = np.expand_dims(thermal_img, axis=0)
        self.detector.allocate_tensors()
        self.detector.set_tensor(self.input_details["index"], thermal_img)
        self.detector.invoke()
        boxes = self.detector.get_tensor(self.output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        classes = self.detector.get_tensor(self.output_details[1]['index'])[0] # Class index of detected objects
        scores = self.detector.get_tensor(self.output_details[2]['index'])[0] # Confidence of detected objects
        num = self.detector.get_tensor(self.output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)
        return boxes, classes, scores, num

    