python save_model.py --model yolov4
import tensorflow as tf
from tensorflow.python.saved_model import tag_constantssaved_model_loaded = tf.saved_model.load(r'.\checkpoints\yolov4-416', tags=[tag_constants.SERVING])
infer = saved_model_loaded.signatures['serving_default']
def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names
  
def format_boxes(bboxes, image_height, image_width):
    for box in bboxes:
        ymin = int(box[0] * image_height)
        xmin = int(box[1] * image_width)
        ymax = int(box[2] * image_height)
        xmax = int(box[3] * image_width)
        width = xmax - xmin
        height = ymax - ymin
        box[0], box[1], box[2], box[3] = xmin, ymin, width, height
    return bboxes
