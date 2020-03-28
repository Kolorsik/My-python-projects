import tensorflow.compat.v1
from imageai.Detection import ObjectDetection
import os

exec_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(exec_path, "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

list = detector.detectObjectsFromImage(input_image=os.path.join(exec_path, "27.jpg"),
                                       output_image_path=os.path.join(exec_path, "new27Img.jpg"),
                                       display_percentage_probability=True,
                                       display_object_name=True)