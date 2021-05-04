import numpy as np
import pickle

from typing import List, Any

from tqdm import tqdm

from simple_object_detection.typing import Image
from simple_object_detection.object import Object
from simple_object_detection.utils.video import StreamSequence


def generate_objects_detections(network: Any,
                                sequence: StreamSequence,
                                mask: Image = None) -> List[List[Object]]:
    """Genera las detecciones de objetos en cada frame de una secuencia de vídeo.

    :param network: red utilizada para la detección de objetos.
    :param sequence: video donde extraer los frames.
    :param mask: máscara para aplicar la zona donde se realizará la detección en la secuencia.
    :return: lista con las detecciones por indexada por frame.
    """
    objects_per_frame = list()
    # Recorrer los frames.
    t = tqdm(total=len(sequence))
    for frame_id, frame in enumerate(sequence):
        # Calcular y extraer los objetos e insertarlos en la lista.
        objects = network.get_objects(frame, mask=mask)
        objects_per_frame.insert(frame_id, objects)
        t.update()
    return objects_per_frame


def save_objects_detections(objects_detections: List[List[Object]], file_output: str) -> None:
    """Guarda las detecciones de objetos en una secuencia.

    Si el archivo existe, lo sobreescribe.

    :param objects_detections: red utilizada para la detección de objetos.
    :param file_output: archivo donde se guardará la lista de detecciones en cada frame.
    """
    with open(file_output, 'wb') as output:
        pickle.dump(objects_detections, output, pickle.HIGHEST_PROTOCOL)


def load_objects_detections(file_path: str, encoding: str = 'ASCII') -> List[Object]:
    """Carga las detecciones guardadas en una archivo.

    :param file_path: dirección al archivo.
    :param encoding: codificación del archivo.
    :return: lista de detecciones de objetos en cada frame.
    """
    with open(file_path, 'rb') as file:
        return pickle.load(file, encoding=encoding)


def filter_objects_by_classes(objects: List[Object], classes: List[str]) -> List[Object]:
    # Preprocesar las clases para ponerlas todas en minúscula.
    classes = [class_name.lower() for class_name in classes]
    # Devolver la lista de los objetos etiquetados con esas clases.
    return list(filter(lambda obj: obj.label.lower() in classes, objects))


def filter_objects_by_min_score(objects: List[Object], min_score: float) -> List[Object]:
    return list(filter(lambda obj: obj.score >= min_score, objects))


def filter_objects_avoiding_duplicated(objects: List[Object],
                                       max_distance: int = 20) -> List[Object]:
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
