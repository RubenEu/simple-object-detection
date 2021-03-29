import cv2
import numpy as np
import pickle

from typing import Tuple, List

from simple_object_detection.typing import Image, Model
from simple_object_detection.object import Object


def filter_objects_by_classes(objects: List[Object], classes: List[str]) -> List[Object]:
    # Preprocesar las clases para ponerlas todas en minúscula.
    classes = [class_name.lower() for class_name in classes]
    # Devolver la lista de los objetos etiquetados con esas clases.
    return list(filter(lambda obj: obj.label in classes, objects))


def filter_objects_by_min_score(objects: List[Object], min_score: float) -> List[Object]:
    return list(filter(lambda obj: obj.score >= min_score, objects))


def filter_objects_avoiding_duplicated(objects: List[Object], max_distance: int) -> List[Object]:
    ...  # TODO


def draw_bounding_boxes(image: Image, objects: List[Object]) -> Image:
    """
    Añade las cajas delimitadoras a todos los objetos en la imagen.

    :param image: ndarray con la imagen (RGB).
    :param objects: ndarray con los objetos.
    :return: imagen con las cajas delimitadoras.
    """
    image_with_boxes = image.copy()
    colors = np.random.uniform(0, 255, size=(len(objects), 3))
    for idx, obj in enumerate(objects):
        (top_left, _, bottom_right, _) = obj.bounding_box
        color = colors[idx]
        label = obj.label
        # Etiqueta para mostrar.
        display_str = "{}: {}%".format(label, int(100 * obj.score))
        # Añadir texto.
        x, y = top_left
        cv2.putText(image_with_boxes, display_str, (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 0.85,
                    color, 2)
        # Añadir caja delimitadora.
        cv2.rectangle(image_with_boxes, top_left, bottom_right, color, 2)
    return image_with_boxes


def load_image(file_path: str) -> Image:
    """
    Carga una imagen en un numpy array en formato RGB.

    :return: imagen.
    """
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_sequence(file_path: str) -> Tuple[int, int, int, List[Image]]:
    """Carga un vídeo como una secuencia de imágenes (RGB).

    :param file_path: ruta del video.
    :return: (anchura, altura, nº de imágenes por segundo, secuencia).
    """
    cap = cv2.VideoCapture(file_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frames_per_second = int(cap.get(cv2.CAP_PROP_FPS))
    frames = list()
    # Decodificar los frames y guardarlos en la lista.
    frames_available = True
    while frames_available:
        retval, frame_bgr = cap.read()
        if retval:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        else:
            frames_available = False
    cap.release()
    return int(width), int(height), frames_per_second, frames


def save_sequence(sequence: List[Image],
                  frame_width: int,
                  frame_height: int,
                  frames_per_second: int,
                  file_output: str) -> None:
    """Guarda una secuencia de frames como un vídeo.

    :param sequence: secuencia de frames.
    :param frame_width: anchura de los frames.
    :param frame_height: altura de los frames.
    :param frames_per_second: frames por segundo.
    :param file_output: archivo donde se guardará (sobreescribe si ya existe).
    """
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(file_output, fourcc, frames_per_second, (frame_width, frame_height))
    for frame in sequence:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()


def save_detections_in_sequence(network: Model, sequence: List[Image], file_output: str) -> None:
    """Guarda las detecciones realizadas en una secuencia.

    :param network: red utilizada para la detección de objetos.
    :param sequence: video donde extraer los frames.
    :param file_output: archivo donde se guardará la lista de detecciones en cada frame.
    """
    objects_per_frame = list()
    # Recorrer los frames.
    for frame_id, frame in enumerate(sequence):
        # Calcular y extraer los objetos e insertarlos en la lista.
        objects = network.get_objects(frame)
        objects_per_frame.insert(frame_id, objects)
    # Guardar las detecciones
    with open(file_output, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(objects_per_frame, output, pickle.HIGHEST_PROTOCOL)


def load_detections_in_sequence(file_input: str) -> List[Object]:
    """Carga las detecciones guardadas en una archivo.

    :param file_input: dirección al archivo.
    :return: lista de detecciones de objetos en cada frame.
    """
    objects_per_frame = None
    with open(file_input, 'rb') as input:
        objects_per_frame = pickle.load(input)
    return objects_per_frame
