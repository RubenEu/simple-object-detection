import tempfile
from abc import ABC, abstractmethod
from typing import AnyStr, List, Any

import cv2

from simple_object_detection.object import Object
from simple_object_detection.typing import Image, Point2D, RelativeBoundingBox
from simple_object_detection.constants import COCO_NAMES


class DetectionModel(ABC):
    """
    Clase abstracta para implementar los modelos de redes neuronales que realizan detección de
    objetos.
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

    def get_output(self, image: Image) -> Any:
        """
        Devuelve la salida de la red neuronal.

        :param image: imagen.
        :return: salida de la red neuronal.
        """
        return self._get_output(image)

    def get_objects(self, image: Image, output: Any = None, mask: Image = None) -> List[Object]:
        """
        Devuelve todos los objetos que se extraen de la salida de la predicción de la red neuronal.

        Realiza la predicción sobre la imagen, o se le puede pasar un output ya calculado sobre el
        que extraer los objetos detectados.

        :param image: imagen.
        :param output: salida de la red neuronal.
        :param mask: máscara para aplicar la zona donde se realizará la detección en la secuencia.
        Si se ha pasado el output también y éste tiene detecciones fuera de la máscara, la máscara
        no será aplicable.
        :return: objetos.
        """
        if output is None:
            # Aplicar máscara a la imagen donde se realizará la detección.
            if mask is not None:
                image = cv2.bitwise_and(image, mask)
            output = self._get_output(image)
        return self._get_objects(image, output)

    def _get_objects(self, image: Image, output: Any) -> List[Object]:
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
    def _load_local(self) -> Any:
        """Método que debe ser implementado para cargar la red de manera offline. Es decir, con
        archivos locales.

        :return: modelo.
        """

    @abstractmethod
    def _load_online(self) -> Any:
        """Método que debe ser implementado para cargar la red de manera online, es decir, sin uso
        de archivos locales.

        :return: modelo.
        """

    @abstractmethod
    def _get_output(self, image: Image) -> Any:
        """Método que debe ser implementado. Debe devolver la salida del modelo que se esté
        implementando. Será introducido por parámetro en los métodos de cálculo de la información
        necesaria para la obtención de los objetos.

        :param image: imagen.
        :return: salida de la red neuronal para la imagen dada.
        """

    @abstractmethod
    def _calculate_number_detections(self, output: Any, *args, **kwargs) -> int:
        """Calcula el número de detecciones obtenidas en la predicción de la red.

        :param output: salida de la red neuronal.
        :param args: argumentos extra que pueden ser necesarios en la implementación del modelo.
        :param kwargs: argumentos extra que pueden ser necesarios en la implementación del modelo.
        :return: cantidad de objetos detectados.
        """

    @abstractmethod
    def _calculate_object_position(self,
                                   output: Any,
                                   obj_id: int,
                                   *args,
                                   **kwargs) -> RelativeBoundingBox:
        """Calcula la posición relativa del objeto en la imagen.

        Este método debe ser implementado y devolver una tupla con el centro en la primera posición,
        y el ancho y alto respectivamente en las dos siguientes posiciones.

        :param output: salida de la red neuronal.
        :param obj_id: índice del objeto en la lista de detecciones del output.
        :param args: argumentos extra que pueden ser necesarios en la implementación del modelo.
        :param kwargs: argumentos extra que pueden ser necesarios en la implementación del modelo.
        :return: centro del objeto, ancho de la caja delimitadora, alto de la caja delimitadora.
        """

    @abstractmethod
    def _calculate_score(self, output: Any, obj_id: int, *args, **kwargs) -> float:
        """Calcula la puntuación del objeto detectado por la red.

        Este método debe ser implementado y devolver la puntuación que ha obtenido el objeto.

        :param output: salida de la red neuronal.
        :param obj_id: índice del objeto en la lista de detecciones del output.
        :param args: argumentos extra que pueden ser necesarios en la implementación del modelo.
        :param kwargs: argumentos extra que pueden ser necesarios en la implementación del modelo.
        :return: puntuación que ha obtenido el objeto.
        """

    @abstractmethod
    def _calculate_label(self, output: Any, obj_id: int, *args, **kwargs) -> str:
        """Calcula cuál es la etiqueta (clase) del objeto.

        Este método debe ser implementado y devolver el nombre de la clase a la que pertenece.

        :param output: salida de la red neuronal.
        :param obj_id: índice del objeto en la lista de detecciones del output.
        :param args: argumentos extra que pueden ser necesarios en la implementación del modelo.
        :param kwargs: argumentos extra que pueden ser necesarios en la implementación del modelo.
        :return: clase a la que pertenece el objeto.
        """


class PyTorchHubModel(DetectionModel, ABC):
    """Clase abstracta para los modelos extraídos de torch-hub.
    """

    def _calculate_number_detections(self, output: Any, *args, **kwargs) -> int:
        return len(output.xyxy[0])

    def _calculate_object_position(self,
                                   output: Any,
                                   obj_id: int,
                                   *args,
                                   **kwargs) -> RelativeBoundingBox:
        xywh = output.xywh[0][obj_id]
        center, width, height = (int(xywh[0]), int(xywh[1])), int(xywh[2]), int(xywh[3])
        return RelativeBoundingBox(Point2D(center[0], center[1]), width, height)

    def _calculate_score(self, output: Any, obj_id: int, *args, **kwargs) -> float:
        return float(output.xywh[0][obj_id][4])

    def _calculate_label(self, output: Any, obj_id: int, *args, **kwargs) -> str:
        class_id = int(output.xywh[0][obj_id][5])
        return COCO_NAMES[class_id]
