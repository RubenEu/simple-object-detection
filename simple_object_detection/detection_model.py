import tempfile
from abc import ABC, abstractmethod
from typing import AnyStr, List, Any

import cv2

from simple_object_detection.constants import COCO_NAMES
from simple_object_detection.object import Object
from simple_object_detection.typing import Image, RelativeBoundingBox, Point2D


class DetectionModel(ABC):
    """Clase abstracta para implementar los modelos de redes neuronales que realizan detección de
    objetos.
    """
    # Carpeta donde se almacenan los modelos locales.
    models_path: str = None
    # Carpeta temporal donde se almacenan los archivos descargados.
    temporal_folder: AnyStr = tempfile.mkdtemp()

    def __init__(self, use_local: bool = False):
        """

        :param use_local: si usar el modelo local. Por defecto usa el modelo online.
        """
        self.offline_mode = use_local
        # Cargar el modelo.
        if use_local:
            # Comprobar que se ha establecido la ruta donde se encuentran los modelos.
            assert self.models_path is not None, 'Set models_path to the directory with the models.'
            self.model = self._load_local()
        else:
            self.model = self._load_online()
        assert self.model is not None, 'The model is not loaded.'

    def get_outputs(self, images: List[Image]) -> List[Any]:
        """Devuelve las salidas de la red neuronal.

        :param images: lista de imágenes.
        :return: lista de las salidas de la red neuronal para cada imagen de entrada.
        """
        return self._get_outputs(images)

    def get_images_objects(self, images: List[Image], mask: Image = None) -> List[List[Object]]:
        """Realiza las detecciones en una lista de imágenes y devuelve las detecciones de los
        objetos obtenidas indexadas por la imagen.

        :param images: lista de imágenes
        :param mask: máscara para aplicar la zona donde se realizará la detección en la secuencia.
        :return: lista de objetos en cada imagen.
        """
        # Aplica la máscara a las imágenes.
        if mask is not None:
            images = [cv2.bitwise_and(image, mask) for image in images]
        # Extrae la salida de la red neuronal.
        outputs = self._get_outputs(images)
        # Extrae la lista de objetos de esa imagen y los devuelve.
        return [self._get_objects(output, image) for image, output in zip(images, outputs)]

    def get_image_objects(self, image: Image, mask: Image = None) -> List[Object]:
        """Devuelve todos los objetos que se extraen de la salida de la predicción de la red
        neuronal.

        :param image: imagen.
        :param mask: máscara para aplicar la zona donde se realizará la detección en la secuencia.
        :return: objetos en la imagen.
        """
        return self.get_images_objects([image], mask)[0]

    def _get_object(self, object_id: int, object_output: Any, image: Image) -> Object:
        """Crea el objeto de la clase ``Object`` con la información pasada por parámetra del output
        de la red neuronal.

        :param object_id: identificador del objeto.
        :param object_output: salida de la red neuronal para el objeto.
        :param image: imagen en la que se detectó el objeto.
        :return: objeto detectado.
        """
        bounding_box_r = self._calculate_object_position(object_output, object_id, image)
        object_detected = Object(
            index=object_id,
            center=bounding_box_r.center,
            width=bounding_box_r.width,
            height=bounding_box_r.height,
            score=self._calculate_score(object_output, object_id),
            label=self._calculate_label(object_output, object_id)
        )
        return object_detected

    def _get_objects(self, output: Any, image: Image) -> List[Object]:
        """Crea los objetos de la clase ``Object`` extraídos del ``output`` de la red neuronal.

        :param output: salida de la red neuronal.
        :param image: imagen donde fueron detectados los objetos.
        :return: lista de objetos extraídos del ``output``.
        """
        return [self._get_object(object_id, object_output, image)
                for object_id, object_output in enumerate(output)]

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
    def _get_outputs(self, images: List[Image]) -> List[Any]:
        """Método que debe ser implementado. Debe devolver las salidas de la red neuronal para cada
         una de las imágenes introducidas. Deben estar indexadas en el mismo orden en el que
         aparecen en la lista ``images``.

        :param images: lista de imágenes.
        :return: salidas de la red neuronal para las imágenes introducidas.
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
                                   object_output: Any,
                                   object_id: int,
                                   image: Image,
                                   *args,
                                   **kwargs) -> RelativeBoundingBox:
        """Calcula la posición relativa del objeto en la imagen.

        Este método debe ser implementado y devolver una tupla con el centro en la primera posición,
        y el ancho y alto respectivamente en las dos siguientes posiciones.

        :param object_output: salida de la red neuronal para un objeto.
        :param object_id: índice del objeto en la lista de detecciones del output.
        :param image: imagen donde fue detectado el objeto. Muchas veces es necesario para obtener
        la posición absoluta.
        :param args: argumentos extra que pueden ser necesarios en la implementación del modelo.
        :param kwargs: argumentos extra que pueden ser necesarios en la implementación del modelo.
        :return: centro del objeto, ancho de la caja delimitadora, alto de la caja delimitadora.
        """

    @abstractmethod
    def _calculate_score(self, object_output: Any, object_id: int, *args, **kwargs) -> float:
        """Calcula la puntuación del objeto detectado por la red.

        Este método debe ser implementado y devolver la puntuación que ha obtenido el objeto.

        :param object_output: salida de la red neuronal para un objeto.
        :param object_id: índice del objeto en la lista de detecciones del output.
        :param args: argumentos extra que pueden ser necesarios en la implementación del modelo.
        :param kwargs: argumentos extra que pueden ser necesarios en la implementación del modelo.
        :return: puntuación que ha obtenido el objeto.
        """

    @abstractmethod
    def _calculate_label(self, object_output: Any, object_id: int, *args, **kwargs) -> str:
        """Calcula cuál es la etiqueta (clase) del objeto.

        Este método debe ser implementado y devolver el nombre de la clase a la que pertenece.

        :param output: salida de la red neuronal para un objeto.
        :param obj_id: índice del objeto en la lista de detecciones del output.
        :param args: argumentos extra que pueden ser necesarios en la implementación del modelo.
        :param kwargs: argumentos extra que pueden ser necesarios en la implementación del modelo.
        :return: clase a la que pertenece el objeto.
        """


class PyTorchHubModel(DetectionModel, ABC):
    """Clase abstracta para los modelos extraídos de torch-hub.
    """
    size: int

    def _get_outputs(self, images: List[Image]) -> List[Any]:
        torch_outputs = self.model(images, size=self.size)
        return [xywh for xywh in torch_outputs.xywh]

    def _calculate_number_detections(self, output: Any, *args, **kwargs) -> int:
        return len(output)

    def _calculate_object_position(self,
                                   object_output: Any,
                                   object_id: int,
                                   image: Image, *args,
                                   **kwargs) -> RelativeBoundingBox:
        center = int(object_output[0]), int(object_output[1])
        width, height = int(object_output[2]), int(object_output[3])
        return RelativeBoundingBox(Point2D(center[0], center[1]), width, height)

    def _calculate_score(self, object_output: Any, object_id: int, *args, **kwargs) -> float:
        return float(object_output[4])

    def _calculate_label(self, object_output: Any, object_id: int, *args, **kwargs) -> str:
        return COCO_NAMES[int(object_output[5])]
