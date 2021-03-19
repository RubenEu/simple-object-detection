import cv2
import numpy as np


def filter_objects(objects, classes=None, min_score=None):
    """
    Filtra una lista de objetos según las propiedades indicadas por parámetro.

    :param objects: lista de objetos para ser filtrados.
    :param classes: lista de clases por las que se desea filtrar.
    :param min_score: puntuación mínima que deben cumplir los objetos.
    :return: array slice de los objetos filtrados.
    """
    if classes:
        objects_classes = [obj.label.lower() for obj in objects]
        filter_classes = [label.lower() for label in classes]
        objects_filter = np.isin(objects_classes, filter_classes)
        objects = objects[objects_filter]
    if min_score:
        objects_filter = [True if obj.is_greatereq_scored_than(min_score) else False for obj in objects]
        objects = objects[objects_filter]
    return objects


def set_bounding_boxes_in_image(image, objects):
    """
    Añade las cajas delimitadoras a todos los objetos en la imagen.

    :param image: ndarray con la imagen (RGB).
    :param objects: ndarray con los objetos.
    :return: imagen con las cajas delimitadoras.
    """
    image_with_boxes = image.copy()
    colors = np.random.uniform(0, 255, size=(len(objects), 3))
    for idx, obj in enumerate(objects):
        (left, right, top, bottom) = obj.bounding_box
        color = colors[idx]
        label = obj.label
        # Etiqueta para mostrar.
        display_str = "{}: {}%".format(label, int(100 * obj.score))
        # Añadir texto.
        cv2.putText(image_with_boxes, display_str, (left, top - 5), cv2.FONT_HERSHEY_COMPLEX, 0.85, color, 2)
        # Añadir caja delimitadora.
        cv2.rectangle(image_with_boxes, (left, top), (right, bottom), color, 2)
    return image_with_boxes


def load_image(file_path):
    """
    Carga una imagen en un numpy array en formato RGB.

    :return: ndarray con la imagen (formato RGB).
    """
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
