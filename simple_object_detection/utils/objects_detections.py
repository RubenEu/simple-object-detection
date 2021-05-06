from math import ceil

import numpy as np
import pickle

from typing import List, Any
from tqdm import tqdm

from simple_object_detection.detection_model import DetectionModel
from simple_object_detection.typing import Image
from simple_object_detection.object import Object
from simple_object_detection.utils.video import StreamSequence


def generate_objects_detections(network: DetectionModel,
                                sequence: StreamSequence,
                                batch_size: int = 1,
                                mask: Image = None) -> List[List[Object]]:
    """Genera las detecciones de objetos en cada frame de una secuencia de vídeo.

    :param network: red utilizada para la detección de objetos.
    :param sequence: video donde extraer los frames.
    :param batch_size: tamaño de frames que se mandan procesar al modelo de detección.
    :param mask: máscara para aplicar la zona donde se realizará la detección en la secuencia.
    :return: lista con las detecciones por indexada por frame.
    """
    frames_objects = []
    iterations = ceil(len(sequence) / batch_size)
    t = tqdm(total=iterations, desc='Generating objects detections')
    for iteration in range(iterations):
        start = iteration * batch_size
        stop = start + batch_size
        frames = [sequence[frame_id] for frame_id in range(start, stop) if frame_id < len(sequence)]
        # frames = sequence[start:stop]  TODO: necesita implementar el slicing en sequence!
        frames_objects += network.get_images_objects(frames, mask)
        t.update()
    return frames_objects


def save_objects_detections(objects_detections: List[List[Object]],
                            file_output: str,
                            pickle_version: int = pickle.DEFAULT_PROTOCOL) -> None:
    """Guarda las detecciones de objetos en una secuencia.

    Si el archivo existe, lo sobreescribe.

    :param objects_detections: red utilizada para la detección de objetos.
    :param file_output: archivo donde se guardará la lista de detecciones en cada frame.
    :param pickle_version: versión del protocolo de pickle.
    """
    with open(file_output, 'wb') as output:
        pickle.dump(objects_detections, output, pickle_version)


def load_objects_detections(file_path: str, encoding: str = 'ASCII') -> List[Object]:
    """Carga las detecciones guardadas en una archivo.

    :param file_path: dirección al archivo.
    :param encoding: codificación del archivo.
    :return: lista de detecciones de objetos en cada frame.
    """
    with open(file_path, 'rb') as file:
        return pickle.load(file, encoding=encoding)


def filter_objects_by_classes(objects: List[Object], classes: List[str]) -> List[Object]:
    """Filtrar los objetos cuya etiqueta está entre las clases especificadas en el parámetro
    ``classes``.

    :param objects: lista de objetos.
    :param classes: lista de nombres de las clases.
    :return: lista de objetos filtrados.
    """
    # Preprocesar las clases para ponerlas todas en minúscula.
    classes = [class_name.lower() for class_name in classes]
    # Devolver la lista de los objetos etiquetados con esas clases.
    return list(filter(lambda obj: obj.label.lower() in classes, objects))


def filter_objects_by_min_score(objects: List[Object], min_score: float) -> List[Object]:
    """Filtrar objetos por puntuación mayor que la indicada por el parámetro ``min_score``

    :param objects: lista de objetos.
    :param min_score: puntuación mínima (incluída) que deben tener los objetos.
    :return: lista de objetos filtrados.
    """
    return list(filter(lambda obj: obj.score >= min_score, objects))


def filter_objects_avoiding_duplicated(objects: List[Object],
                                       max_distance: int = 20) -> List[Object]:
    """Filtra los objetos evitando aquellas posibles que sean detecciones múltiples.

    El fundamento del algoritmo es que si se detectan dos objetos con un centroide muy cercano, a
    una distancia máxima indicada por ``max_distance``, entonces es una detección múltiple.

    El conflicto se resuelve eliminando las detecciones múltiple y escogiendo la que mejor
    puntuación ha obtenido en la detección.

    :param objects: lista de objetos.
    :param max_distance: máxima distancia entre centros para considerar que ese objeto puede ser
    un duplicado.
    :return: lista de objetos filtrados.
    """
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


def filter_objects_inside_mask_region(objects: List[Object], mask: Image) -> List[Object]:
    """Filtra los objetos que están dentro de una máscara.

    Se toma como punto de referencia del objeto su centroide.

    :param objects: lista de objetos.
    :param mask: máscara con la región dónde se filtrarán los vehículos.
    :return: lista de objetos filtrados.
    """
    return [object_ for object_ in objects if mask[object_.center[1], object_.center[0]].all()]
