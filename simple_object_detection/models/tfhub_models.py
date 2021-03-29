import tensorflow as tf
import numpy as np
from abc import ABC
from ..detection_model import DetectionModel
from ..object import Object


class TFHubModel(DetectionModel, ABC):
    """
    Clase abstracta común para los modelos extraídos de tensorflow hub.
    
    Los modelos para las redes implementadas con esta clase se pueden encontrar en:
        - https://tfhub.dev/s?module-type=image-object-detection
        - https://tfhub.dev/tensorflow/collections/object_detection/1
    
    Uso local: para su correcto funcionamiento tendrá que descomprimirse el archivo .tar.gz en
    una carpeta con el mismo nombre que el archivo (como convención).
    """

    def __init__(self, *args, **kwargs):
        DetectionModel.__init__(self, *args, **kwargs)

    def _preprocess_image(self, image):
        return tf.image.convert_image_dtype(image, tf.float32)[tf.newaxis, ...]

    def _get_output(self, image):
        input_pattern = self._preprocess_image(image)
        output = self.detector(input_pattern)
        return output

    def _calculate_number_detections(self, output, *args, **kwargs):
        return output['detection_boxes'].shape[0]

    def _calculate_bounding_box(self, output, obj_id, *args, **kwargs):
        output_box = output['detection_boxes'][obj_id]
        image = kwargs['image']
        im_height, im_width = image.shape[0:2]
        ymin, xmin, ymax, xmax = output_box.numpy()
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
        return np.array([left, right, top, bottom], dtype=np.uint32)

    def _calculate_score(self, output, obj_id, *args, **kwargs):
        output_score = output['detection_scores'][obj_id]
        return output_score.numpy()

    def _calculate_label(self, output, obj_id, *args, **kwargs):
        output_label = output['detection_class_entities'][obj_id]
        return output_label.numpy().decode()
