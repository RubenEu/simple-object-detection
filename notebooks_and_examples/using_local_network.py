from simple_object_detection import DetectionModel
from simple_object_detection.models import *
from simple_object_detection.utils import *

image = load_image(file_path='../sample_data/tf.jpg')
# Configurar la ruta de los modelos (local).
DetectionModel.models_path = '../models_data/'

# Ejemplo usando YOLOv3 (la más rápida, por el momento).
net = SSDMobileNetV2(use_local=True)
output = net.get_output(image)
objects = net.get_objects(image, output)
image_with_boxes = set_bounding_boxes_in_image(image, objects)

# Mostrar con cv2.
# Hace un redimensionado y cambia a BGR ya que opencv usa BGR por defecto.
cv2.imshow('Image',
           cv2.resize(
               cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR),
               (int(image_with_boxes.shape[1] * 4/5), int(image_with_boxes.shape[0] * 4/5))
           ))
# Esperar a pulsar escape para cerrar la ventana.
cv2.waitKey(0)
