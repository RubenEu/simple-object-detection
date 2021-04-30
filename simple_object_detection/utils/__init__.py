import cv2
import numpy as np
import pickle

from typing import Tuple, List, Union

from simple_object_detection.typing import Image, Model, SequenceLoaded
from simple_object_detection.object import Object
from simple_object_detection.exceptions import SimpleObjectDetectionException


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
    cap = cv2.VideoCapture(file_path)
    # Comprobar si el vídeo está disponible.
    if not cap.isOpened():
        raise Exception(f'The {file_path} can\'t be opened or doesn\'t exists.')
    # Parámetros de la secuencia.
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frames_per_second = float(cap.get(cv2.CAP_PROP_FPS))
    frames = list()
    timestamps = list()
    # Decodificar los frames y guardarlos en la lista.
    frames_available = True
    frame_id = 0
    start_cond = True
    end_cond = True
    while frames_available and end_cond:
        retval, frame_bgr = cap.read()
        # Comprobar si hay rango, y en ese caso, si el frame que se ha leído está en el rango.
        start_cond = frame_start is None or (frame_start is not None and frame_start <= frame_id)
        end_cond = frame_end is None or (frame_end is not None and frame_id < frame_end)
        range_cond = start_cond and end_cond
        # Comprobar si se cargó el frame siguiente.
        if retval:
            # Comprobar que el frame está en el rango.
            if range_cond:
                # Convertir a RGB.
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                # Guardar el frame.
                frames.append(frame_rgb)
                # Añadir el timestamp correspondiente a este frame.
                timestamps.append(int(cap.get(cv2.CAP_PROP_POS_MSEC)))
            # Pasar al siguiente frame id.
            frame_id += 1
        else:
            frames_available = False
    cap.release()
    return int(width), int(height), frames_per_second, frames, timestamps


def change_frame_rate_sequence(sequence: List[Image],
                               fps: float,
                               new_frame_rate: float) -> List[Image]:
    sequence_frames = len(sequence)
    sequence_duration = sequence_frames / fps
    sequence_new_frames = int(sequence_duration * new_frame_rate)
    frame_step = sequence_frames / (sequence_duration * new_frame_rate)
    new_sequence: List[Image] = list()
    for frame_index in range(sequence_new_frames):
        new_frame_index = int(frame_step * frame_index)
        # Asegurar que no salta fuera de la secuencia original.
        if new_frame_index < sequence_frames:
            new_sequence.append(sequence[new_frame_index])
    return new_sequence


def save_sequence(sequence: List[Image],
                  frame_width: int,
                  frame_height: int,
                  frames_per_second: float,
                  file_output: str,
                  resize_factor: float = 1,
                  new_frame_rate: float = None) -> None:
    """Guarda una secuencia de frames como un vídeo.

    :param sequence: secuencia de frames.
    :param frame_width: anchura de los frames.
    :param frame_height: altura de los frames.
    :param frames_per_second: frames por segundo.
    :param file_output: archivo donde se guardará (sobreescribe si ya existe).
    :param resize_factor: la salida tendrá un tamaño redimensionado por el factor indicado.
    :param new_frame_rate: nueva tasa de frames por segundo (tiene que ser menor que la tasa original).
    """
    # Redimensionar el frame si es necseario.
    frame_width = int(frame_width * resize_factor)
    frame_height = int(frame_height * resize_factor)
    # Cáculo del nuevo frame rate.
    if new_frame_rate is not None:
        sequence = change_frame_rate_sequence(sequence, frames_per_second, new_frame_rate)
        frames_per_second = new_frame_rate
    # Cargar codec y video de salida.
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(file_output, fourcc, frames_per_second, (frame_width, frame_height))
    # Procesar cada frame de la secuencia.
    for frame in sequence:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if resize_factor != 1:
            frame = cv2.resize(frame, (frame_width, frame_height))
        out.write(frame)
    out.release()


def generate_detections_in_sequence(network: Model, sequence: List[Image],
                                    mask: Image = None) -> List[List[Object]]:
    """Genera las detecciones de objetos en cada frame.

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


def save_detections_in_sequence(network: Model, sequence: List[Image], file_output: str,
                                mask: Image = None) -> None:
    """Guarda las detecciones realizadas en una secuencia.

    :param network: red utilizada para la detección de objetos.
    :param sequence: video donde extraer los frames.
    :param file_output: archivo donde se guardará la lista de detecciones en cada frame.
    :param mask: máscara para aplicar la zona donde se realizará la detección en la secuencia.
    """
    objects_per_frame = generate_detections_in_sequence(network, sequence, mask)
    # Guardar las detecciones
    with open(file_output, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(objects_per_frame, output, pickle.HIGHEST_PROTOCOL)


def load_detections_in_sequence(file_path: str) -> List[Object]:
    """Carga las detecciones guardadas en una archivo.

    :param file_path: dirección al archivo.
    :return: lista de detecciones de objetos en cada frame.
    """
    with open(file_path, 'rb') as file:
        return pickle.load(file)


def load_sequence_by_frames(frames_files: List[str],
                            fps: float,
                            width: int = None,
                            height: int = None) -> SequenceLoaded:
    """
    Carga una lista de imágenes dada la dirección del archivo y genera una secuencia de vídeo.

    Calcula la duración de la secuencia con: duración = frames / fps.

    Calcula el tiempo interpolando.
    :param frames_files: direcciones de los archivos a las frames (ordenados).
    :param fps: fotogramas por segundo.
    :param width: ancho de la secuencia de video.
    :param height: altura de la secuencia de video.
    :return: (anchura, altura, nº de imágenes por segundo, secuencia, timestamps).
    """
    duration_ms: int = int((len(frames_files) / fps) * 1000)
    ms_per_frame = duration_ms / len(frames_files)
    timestamps: List[int] = [int(i*ms_per_frame) for i in range(len(frames_files))]
    frames: List[Image] = list()
    for frame_file in frames_files:
        frames.append(load_image(frame_file))
    # Calcular el ancho y alto dado el primer frame.
    if width is None:
        width = frames[0].shape[1]
    if height is None:
        height = frames[0].shape[0]

    return width, height, fps, frames, timestamps
