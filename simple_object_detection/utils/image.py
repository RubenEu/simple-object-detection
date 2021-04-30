import numpy as np
import cv2

from typing import List

from simple_object_detection.typing import Image
from simple_object_detection.object import Object
from simple_object_detection.exceptions import SimpleObjectDetectionException


def draw_bounding_boxes(image: Image, objects: List[Object]) -> Image:
    """Añade las cajas delimitadoras a todos los objetos en la imagen.

    :param image: ndarray con la imagen (RGB).
    :param objects: ndarray con los objetos.
    :return: imagen con las cajas delimitadoras.
    """
    image_with_boxes = image.copy()
    colors = np.random.uniform(0, 255, size=(len(objects), 3))
    for idx, obj in enumerate(objects):
        (top_left, top_right, bottom_right, bottom_left) = obj.bounding_box
        color = colors[idx]
        label = obj.label
        # Etiqueta para mostrar.
        display_str = "{}: {}%".format(label, int(100 * obj.score))
        # Añadir texto.
        x, y = top_left
        cv2.putText(image_with_boxes, display_str, (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 0.85,
                    color, 2)
        # Añadir caja delimitadora.
        cv2.line(image_with_boxes, top_left, top_right, color, 2, cv2.LINE_AA)
        cv2.line(image_with_boxes, top_right, bottom_right, color, 2, cv2.LINE_AA)
        cv2.line(image_with_boxes, bottom_right, bottom_left, color, 2, cv2.LINE_AA)
        cv2.line(image_with_boxes, bottom_left, top_left, color, 2, cv2.LINE_AA)
    return image_with_boxes


def load_image(file_path: str) -> Image:
    """Carga una imagen en un numpy array en formato RGB.

    :return: imagen.
    """
    img = cv2.imread(file_path)
    if img is None:
        raise SimpleObjectDetectionException('The image path doesn\'t exists.')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
