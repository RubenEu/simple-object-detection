import cv2
import numpy as np
import pickle


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


def load_sequence(file_path):
    """Carga un vídeo como una secuencia de imágenes (RGB).

    :param file_path: ruta del video.
    :return: lista con los frames.
    """
    cap = cv2.VideoCapture(file_path)
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
    return frames


def save_sequence(sequence, frame_width, frame_height, frames_per_second, file_output):
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(file_output, fourcc, frames_per_second, (frame_width, frame_height))
    for frame in sequence:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()


def save_detections_in_sequence(network, sequence, file_output):
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


def load_detections_in_sequence(file_input):
    """Carga las detecciones guardadas en una archivo.

    :param file_input: dirección al archivo.
    :return: lista de detecciones de objetos en cada frame.
    """
    objects_per_frame = None
    with open(file_input, 'rb') as input:
        objects_per_frame = pickle.load(input)
    return objects_per_frame
