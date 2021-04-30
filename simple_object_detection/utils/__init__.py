import cv2
import numpy as np
import pickle

from typing import Tuple, List, Union

from simple_object_detection.typing import Image, Model, SequenceLoaded
from simple_object_detection.object import Object
from simple_object_detection.exceptions import SimpleObjectDetectionException
from simple_object_detection.utils.video import Sequence


def filter_objects_by_classes(objects: List[Object], classes: List[str]) -> List[Object]:
    # Preprocesar las clases para ponerlas todas en minúscula.
    classes = [class_name.lower() for class_name in classes]
    # Devolver la lista de los objetos etiquetados con esas clases.
    return list(filter(lambda obj: obj.label.lower() in classes, objects))


def filter_objects_by_min_score(objects: List[Object], min_score: float) -> List[Object]:
    return list(filter(lambda obj: obj.score >= min_score, objects))


def filter_objects_avoiding_duplicated(objects: List[Object], max_distance: int = 20) -> List[Object]:
    # Lista de las posiciones en 'objects' de los objetos eliminados.
    removed_objects_id = list()
    # Buscar los posibles candidatos para cada objeto.
    for obj_id, obj_detection in enumerate(objects):
        for candidate_id, candidate_detection in enumerate(objects):
            # Ignorar el mismo objeto como posible candidato.
            if obj_id == candidate_id:
                continue
            # Ignorar si alguno de los que se está comparando ha sido eliminado ya.
            if obj_id in removed_objects_id or candidate_id in removed_objects_id:
                continue
            # Calcular la distancia euclídea entre ambas detecciones.
            p = np.array(obj_detection.center)
            q = np.array(candidate_detection.center)
            distance = np.linalg.norm(p - q)
            # Si hay poca distancia, puede ser el mismo objeto.
            if distance <= max_distance:
                # Eliminar el que menos puntuación tiene.
                if obj_detection.score > candidate_detection.score:
                    removed_objects_id.append(candidate_id)
                else:
                    removed_objects_id.append(obj_id)
    # Lista de los objetos que han pasado el filtro.
    objects_filtered: List[Object] = list()
    for obj_id, obj_detection in enumerate(objects):
        if obj_id not in removed_objects_id:
            objects_filtered.append(obj_detection)
    return objects_filtered


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
    """
    Carga una imagen en un numpy array en formato RGB.

    :return: imagen.
    """
    img = cv2.imread(file_path)
    if img is None:
        raise SimpleObjectDetectionException('The image path doesn\'t exists.')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_sequence(file_path: str,
                  frame_start: Union[None, int] = None,
                  frame_end: Union[None, int] = None) -> SequenceLoaded:
    """Carga un vídeo como una secuencia de imágenes (RGB).

    :param file_path: ruta del video.
    :param frame_start: indica a partir de qué frame (incluído) que cargará.
    :param frame_end: indica hasta qué frame (no incluído) se cargará.
    :return: (anchura, altura, nº de imágenes por segundo, secuencia, timestamps).
    """
    raise Exception('Deprecated. Use simple_object_detection.utils.video.Sequence class instead.')


def save_sequence(sequence: Sequence, file_output: str) -> None:
    """Guarda una secuencia de frames como un vídeo.

    :param sequence: secuencia de frames.
    :param file_output: archivo donde se guardará (sobreescribe si ya existe).
    """
    # Redimensionar el frame si es necseario.
    frame_width = sequence.width
    frame_height = sequence.height
    # Cargar codec y video de salida.
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(file_output, fourcc, sequence.fps, (frame_width, frame_height))
    # Guardar cada frame de la secuencia.
    for frame in sequence:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    out.release()


def generate_objects_detections(network: Model,
                                sequence: Sequence,
                                mask: Image = None) -> List[List[Object]]:
    """Genera las detecciones de objetos en cada frame de una secuencia de vídeo.

    :param network: red utilizada para la detección de objetos.
    :param sequence: video donde extraer los frames.
    :param mask: máscara para aplicar la zona donde se realizará la detección en la secuencia.
    :return: lista con las detecciones por indexada por frame.
    """
    objects_per_frame = list()
    # Recorrer los frames.
    for frame_id, frame in enumerate(sequence):
        # Calcular y extraer los objetos e insertarlos en la lista.
        objects = network.get_objects(frame, mask=mask)
        objects_per_frame.insert(frame_id, objects)
    return objects_per_frame


def save_objects_detections(objects_detections: List[List[Object]], file_output: str) -> None:
    """Guarda las detecciones de objetos en una secuencia.

    Si el archivo existe, lo sobreescribe.

    :param objects_detections: red utilizada para la detección de objetos.
    :param file_output: archivo donde se guardará la lista de detecciones en cada frame.
    """
    with open(file_output, 'wb') as output:
        pickle.dump(objects_detections, output, pickle.HIGHEST_PROTOCOL)


def load_object_detections(file_path: str) -> List[Object]:
    """Carga las detecciones guardadas en una archivo.

    :param file_path: dirección al archivo.
    :return: lista de detecciones de objetos en cada frame.
    """
    with open(file_path, 'rb') as file:
        return pickle.load(file)
