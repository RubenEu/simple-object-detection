from abc import ABC, abstractmethod
import tempfile


class DetectionModel(ABC):
    """
    Clase abstracta para implementar los modelos de redes neuronales que realizan detección de objetos.
    """
    # Carpeta donde se almacenan los modelos locales.
    models_path = None
    # Carpeta temporal donde se almacenan los archivos descargados.
    temporal_folder = tempfile.mkdtemp()

    def __init__(self, use_local=False):
        """
        Clase abstracta para implementarla en los modelos de redes neuronales.
        
        Para uso local de las redes se buscará en la carpeta configurada en el atributo de clase 'models_path'.
        Además en la clase de cada red se indicaré la carpeta o archivos necesarios para su correcto funcionamiento.

        Para una correcta implementación se necesitan sobreescribir los siguientes métodos:

        - _load_local(self): carga el modelo desde la carpeta de modelos local.

        - _load_online(self): carga el modelo descargándolo previamente.

        - _get_output(self, image): tendrá como entrada la imagen sin preprocesar. Debe devolver la salida de la red
            neuronal. El formato de salida será el que cada red utilice.

        - _calculate_number_detections(self, output, *args, **kwargs): este método será llamado para obtener el número
            de detecciones que ha realizado la red. Su salida será un número entero.

        - _calculate_bounding_box(self, output, obj_id, *args, **kwargs): recibe como parámetro la salida y la posición
            del objeto. La salida debe ser del siguiente formato: ndarray 4 posiciones de tipo uint32 donde se indiquen
            4 enteros indicando la posición [left, right, top, bottom], o lo que es lo mismo, la posición mínima de la
            izquierda, la posición máxima de la derecha, la posición más alta y la más baja de la caja delimitadora.

        - _calculate_score(self, output, obj_id, *args, **kwargs): recibe la salida de la red y el id del objeto y debe
            devolver la puntuación que ha obtenido ese objeto.

        - _calculate_label(self, output, obj_id, *args, **kwargs): recibe la salida de la red y el id del objeto y debe
            devolver el string de la etiqueta de la clase a la que pertenece el objeto.

        :param use_local: (opcional)parámetro para indicar si usar los modelos localmente o descargarlos (en caso de
            estar implementado).
        """
        self.offline_mode = use_local
        # Comprobar que se ha establecido la ruta donde se encuentran los modelos localmente.
        if use_local:
            assert self.models_path is not None, 'Set models_path to the directory with the models.'
            self._load_local()
        else:
            self._load_online()

    def get_output(self, image):
        """
        Devuelve la salida de la red neuronal.

        :param image: imagen ndarray uint8.
        :return: salida de la red neuronal.
        """
        return self._get_output(image)

    def get_objects(self, image, output=None):
        """
        Devuelve todos los objetos que se extraen de la salida (output).

        Realiza la predicción sobre la imagen, o se le puede pasar un output ya calculado sobre el que extrae los
        objetos detectados.

        :param image: imagen ndarray uint8.
        :param output: (opcional) salida de la red neuronal.
        :return: ndarray de objetos.
        """
        if not output:
            output = self._get_output(image)
        return self._get_objects(image, output)

    @abstractmethod
    def _load_local(self):
        return None

    @abstractmethod
    def _load_online(self):
        return None

    @abstractmethod
    def _get_output(self, image):
        return None

    @abstractmethod
    def _get_objects(self, image, output):
        return None

    @abstractmethod
    def _calculate_number_detections(self, output, *args, **kwargs):
        return None

    @abstractmethod
    def _calculate_bounding_box(self, output, obj_id, *args, **kwargs):
        return None

    @abstractmethod
    def _calculate_score(self, output, obj_id, *args, **kwargs):
        return None

    @abstractmethod
    def _calculate_label(self, output, obj_id, *args, **kwargs):
        return None
