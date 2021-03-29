import tempfile
import tensorflow as tf

from abc import ABC, abstractmethod
from typing import AnyStr, List, Tuple

from simple_object_detection.object import Object
from simple_object_detection.typing import Model, ModelOutput, Image, Point2D
from simple_object_detection.constants import COCO_NAMES


class DetectionModel(ABC):
    """
    Clase abstracta para implementar los modelos de redes neuronales que realizan detección de
    bjetos.
    """
    # Carpeta donde se almacenan los modelos locales.
    models_path: str = None
    # Carpeta temporal donde se almacenan los archivos descargados.
    temporal_folder: AnyStr = tempfile.mkdtemp()

    def __init__(self, use_local: bool = False):
        self.offline_mode = use_local
        # Cargar el modelo.
        if use_local:
            # Comprobar que se ha establecido la ruta donde se encuentran los modelos.
            assert self.models_path is not None, 'Set models_path to the directory with the models.'
            self.model = self._load_local()
        else:
            self.model = self._load_online()
        assert self.model is not None, 'The model is not loaded.'

    def get_output(self, image: Image) -> ModelOutput:
        """
        Devuelve la salida de la red neuronal.

        :param image: imagen.
        :return: salida de la red neuronal.
        """
        return self._get_output(image)

    def get_objects(self,
                    image: Image,
                    output: ModelOutput = None) -> List[Object]:
        """
        Devuelve todos los objetos que se extraen de la salida de la predicción de la red neuronal.

        Realiza la predicción sobre la imagen, o se le puede pasar un output ya calculado sobre el
        que extraer los objetos detectados.

        :param image: imagen.
        :param output: salida de la red neuronal.
        :return: objetos.
        """
        if not output:
            output = self._get_output(image)
        return self._get_objects(image, output)

    def _get_objects(self, image: Image, output: ModelOutput) -> List[Object]:
        """
        Realiza la extracción de todos los objetos que se encuentran en la salida de la red neuronal
        pasada por parámetro.

        Este método puede ser sobreescrito si el comportamiento del modelo implementado es diferente
        o requiere de modificaciones especiales.

        :param image: imagen.
        :param output: salida de la red neuronal.
        :return: objetos.
        """
        objects = list()
        num_detections = self._calculate_number_detections(output)
        # Añadir los objetos a la lista.
        for obj_id in range(0, num_detections):
            center, width, height = self._calculate_object_position(output, obj_id, image=image)
            object_detected = Object(
                index=obj_id,
                center=center,
                width=width,
                height=height,
                score=self._calculate_score(output, obj_id),
                label=self._calculate_label(output, obj_id)
            )
            objects.append(object_detected)
        return objects

    @abstractmethod
    def _load_local(self) -> Model:
        ...

    @abstractmethod
    def _load_online(self) -> Model:
        """
        Método que debe ser implementado para cargar la red de manera online, es decir, sin uso de
        archivos locales.

        :return: modelo.
        """
        ...

    @abstractmethod
    def _get_output(self, image: Image) -> ModelOutput:
        """
        Método que debe ser implementado. Debe devolver la salida del modelo que se esté
        implementando. Será introducido por parámetro en los métodos de cálculo de la información
        necesaria para la obtención de los objetos.

        :param image: imagen.
        :return: salida de la red neuronal para la imagen dada.
        """
        ...

    @abstractmethod
    def _calculate_number_detections(self, output: ModelOutput, *args, **kwargs) -> int:
        """
        Calcula el número de detecciones obtenidas en la predicción de la red.

        :param output: salida de la red neuronal.
        :param args: argumentos extra que pueden ser necesarios en la implementación del modelo.
        :param kwargs: argumentos extra que pueden ser necesarios en la implementación del modelo.
        :return: cantidad de objetos detectados.
        """
        ...

    @abstractmethod
    def _calculate_object_position(self,
                                   output: Model,
                                   obj_id: int,
                                   *args,
                                   **kwargs) -> Tuple[Point2D, int, int]:
        """
        Calcula la posición relativa del objeto en la imagen.

        Este método debe ser implementado y devolver una tupla con el centro en la primera posición,
        y el ancho y alto respectivamente en las dos siguientes posiciones.

        :param output: salida de la red neuronal.
        :param obj_id: índice del objeto en la lista de detecciones del output.
        :param args: argumentos extra que pueden ser necesarios en la implementación del modelo.
        :param kwargs: argumentos extra que pueden ser necesarios en la implementación del modelo.
        :return: centro del objeto, ancho de la caja delimitadora, alto de la caja delimitadora.
        """
        ...

    @abstractmethod
    def _calculate_score(self, output: ModelOutput, obj_id: int, *args, **kwargs) -> float:
        """
        Calcula la puntuación del objeto detectado por la red.

        Este método debe ser implementado y devolver la puntuación que ha obtenido el objeto.

        :param output: salida de la red neuronal.
        :param obj_id: índice del objeto en la lista de detecciones del output.
        :param args: argumentos extra que pueden ser necesarios en la implementación del modelo.
        :param kwargs: argumentos extra que pueden ser necesarios en la implementación del modelo.
        :return: puntuación que ha obtenido el objeto.
        """
        ...

    @abstractmethod
    def _calculate_label(self, output: ModelOutput, obj_id: int, *args, **kwargs) -> str:
        """
        Calcula cuál es la etiqueta (clase) del objeto.

        Este método debe ser implementado y devolver el nombre de la clase a la que pertenece.

        :param output: salida de la red neuronal.
        :param obj_id: índice del objeto en la lista de detecciones del output.
        :param args: argumentos extra que pueden ser necesarios en la implementación del modelo.
        :param kwargs: argumentos extra que pueden ser necesarios en la implementación del modelo.
        :return: clase a la que pertenece el objeto.
        """
        ...


class PyTorchHubModel(DetectionModel, ABC):
    """
    Clase abstracta para los modelos extraídos de torch-hub.
    """

    def _calculate_number_detections(self, output: ModelOutput, *args, **kwargs) -> int:
        return len(output.xyxy[0])

    def _calculate_object_position(self, output: Model,
                                   obj_id: int, *args, **kwargs) -> Tuple[Point2D, int, int]:
        xywh = output.xywh[0][obj_id]
        center, width, height = (xywh[0], xywh[1]), xywh[2], xywh[3]
        return center, width, height

    def _calculate_score(self, output: ModelOutput, obj_id: int, *args, **kwargs) -> float:
        return float(output.xywh[0][obj_id][4])

    def _calculate_label(self, output: ModelOutput, obj_id: int, *args, **kwargs) -> str:
        class_id = int(output.xywh[0][obj_id][5])
        return COCO_NAMES[class_id]


class TFHubModel(DetectionModel, ABC):
    """
    Clase abstracta común para los modelos extraídos de tensorflow hub.

    Los modelos para las redes implementadas con esta clase se pueden encontrar en:
        - https://tfhub.dev/s?module-type=image-object-detection
        - https://tfhub.dev/tensorflow/collections/object_detection/1

    Uso local: para su correcto funcionamiento tendrá que descomprimirse el archivo .tar.gz en
    una carpeta con el mismo nombre que el archivo (como convención).
    """

    @staticmethod
    def _preprocess_image(image: Image) -> Image:
        """Preprocesa la imagen para ser introducida en la red.

        :param image: imagen.
        :return: imagen preprocesada.
        """
        return tf.image.convert_image_dtype(image, tf.float32)[tf.newaxis, ...]

    def _get_output(self, image):
        input_pattern = self._preprocess_image(image)
        output = self.model(input_pattern)
        return output

    def _calculate_number_detections(self, output: ModelOutput, *args, **kwargs) -> int:
        return output['detection_boxes'].shape[0]

    def _calculate_object_position(self, output: Model,
                                   obj_id: int, *args, **kwargs) -> Tuple[Point2D, int, int]:
        im_height, im_width = kwargs['image'].shape[0:2]
        ymin, xmin, ymax, xmax = output['detection_boxes'][obj_id].numpy()
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height,
                                      ymax * im_height)
        center = int(left + (right - left) / 2), int(top + (bottom - top) / 2)
        width, height = int(right - left), int(bottom - top)
        return center, width, height

    def _calculate_score(self, output: ModelOutput, obj_id: int, *args, **kwargs) -> float:
        return float(output['detection_scores'][obj_id])

    def _calculate_label(self, output: ModelOutput, obj_id: int, *args, **kwargs) -> str:
        return output['detection_class_entities'][obj_id].numpy().decode()
