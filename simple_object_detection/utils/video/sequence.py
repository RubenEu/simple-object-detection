from typing import List, Tuple, Optional

import cv2

from simple_object_detection.exceptions import SimpleObjectDetectionException
from simple_object_detection.typing import Image


class StreamSequence:
    """Clase para cargar los frames de una secuencia de vídeo.

    Se utiliza la notación de acceso a un objeto ``object[item]``. Así se puede ir cargando
    el vídeo poco a poco sin llegar a saturar la memoria RAM.

    TODO:
    - Ir marcando los que se han visto.
    - Crear un hilo que vaya trayendo los nuevos a memoria.
    - Etc. Etc. Optimizar esto!
    - Implementar __iter__. (PEP 234)
    """
    def __init__(self, video_path: str, cache_size: int = 100):
        # Abrir el stream con OpenCV.
        self.stream = self._open_video_stream(video_path)
        # Información del vídeo.
        self.width: int = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height: int = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps: float = float(self.stream.get(cv2.CAP_PROP_FPS))
        self._num_frames_available: int = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        # Caching (Almacena el número del frame y la imagen).
        self._cache: List[Tuple[int, Image]] = [(..., ...)] * cache_size
        # Inicio y fin del vídeo.
        self._start_frame = 0
        self._end_frame = self._num_frames_available - 1

    def __del__(self) -> None:
        """Libera el recurso del vídeo cargado y elimina el objeto también.
        """
        self.stream.release()
        del self.stream

    def __getitem__(self, item: int) -> Image:
        """
        TODO: Cachear. Cargar chunk. Etc. Etc. Mostrar rendimiento haciendo caching.
        https://medium.com/fintechexplained/advanced-python-how-to-implement-caching-in-python-application-9d0a4136b845
        TODO: Añadir slicing:
        https://www.geeksforgeeks.org/implementing-slicing-in-__getitem__/#:~:text=slice%20is%20a%20constructor%20in,can%20be%20defined%20inside%20it.&text=Parameter%3A,constructor%20to%20create%20slice%20object.
        """
        # Comprobación del ítem.
        if not isinstance(item, int):
            raise TypeError()
        # Calcular el índice del frame.
        fid = self._calculate_frame_index(item)
        # Extraer el frame buscado.
        frame_bgr = self._get_frame(fid)
        # Comprobar el valor de salida.
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return frame_rgb

    def __len__(self) -> int:
        return self.num_frames_available

    def set_start_frame(self, frame: int) -> None:
        """Establece el frame inicial.

        Realiza la comprobación de que sea superior a 0 y que no sea superior o igual al frame
        final.

        :param frame: frame inicial.
        :return: None.
        """
        if frame < 0:
            raise SimpleObjectDetectionException('No se puede establecer un frame inicial inferior'
                                                 'a 0.')
        elif frame >= self._end_frame:
            raise SimpleObjectDetectionException('No se puede establecer un frame igual o superior'
                                                 'al frame final.')
        self._start_frame = frame

    def set_end_frame(self, frame: int) -> None:
        """Establece el frame final.

        Realiza la comprobación de que no sea superior o igual a la cantidad de frames disponibles.
        Además comprueba también que no sea inferior o igual al frame inicial.

        :param frame: frame final.
        :return: None.
        """
        if frame >= self._num_frames_available:
            raise SimpleObjectDetectionException('No se puede establecer un frame final superior o'
                                                 'igual a la cantidad de frames disponibles.')
        elif frame <= self._start_frame:
            raise SimpleObjectDetectionException('No se puede establecer un frame final menor o'
                                                 'igual que el frame inicial.')
        self._end_frame = frame

    def _calculate_frame_index(self, fid: int) -> int:
        """Si se ha establecido un límite inferior o superior distinto, recalcula.

        Comprueba que el índice está entre 0 y el número de frames disponibles.

        :param fid:
        :return: índice calculado del frame.
        """
        calculated_fid = self._start_frame + fid
        # Comprobar que está en el intervalo especificado.
        if not self._start_frame <= calculated_fid <= self._end_frame:
            raise IndexError(f'El frame {calculated_fid} está fuera del intervalo'
                             f'[{self._start_frame}, {self._end_frame}].')
        return calculated_fid

    def _get_frame(self, fid: int) -> Image:
        """TODO: Documentar.

        :param fid:
        :return:
        """
        # Comprobación del índice.
        if fid >= self.num_frames_available:
            raise IndexError('Se ha excedido el límite.')
        # Buscar en caché primero.
        cached = self._search_in_cache(fid)
        if cached is None:
            return self._pull_to_cache(fid)
        return cached

    def _search_in_cache(self, fid: int) -> Optional[Image]:
        """TODO: Documentar.

        :param fid:
        :return:
        """
        expected_index = fid % len(self._cache)
        frame_id, frame = self._cache[expected_index]
        if frame_id == fid:
            return frame
        return None

    def _pull_to_cache(self, fid: int) -> Image:
        """Trae a caché los siguientes frames y devuelve el buscado.

        TODO: Documentar.

        :return:
        """
        # Establecer en el frame que se busca.
        retval = self.stream.set(cv2.CAP_PROP_POS_FRAMES, float(fid))
        if not retval:
            raise Exception('Ocurrió un error al posicionar el número de frame.')
        # Iterar mientras haya frames disponibles.from
        actual_frame_id = fid
        # Añadir los siguientes frames que quepan la caché.
        while actual_frame_id < self.num_frames_available:
            # Capturar frame a frame.
            actual_frame_id = int(self.stream.get(cv2.CAP_PROP_POS_FRAMES))
            ret, frame = self.stream.read()
            # Comprobar si se ha leído el frame correctamente.
            if not ret:
                break
            # Añadir a la caché
            self._cache[actual_frame_id % len(self._cache)] = (actual_frame_id, frame)
            # Comprobar si se ha rellenado la caché.
            if (actual_frame_id + 1) % len(self._cache) == 0:
                break
        # Devolver el frame que se buscaba.
        return self._cache[fid % len(self._cache)][1]

    @staticmethod
    def _open_video_stream(video_path: str) -> cv2.VideoCapture:
        """Abre el streaming del vídeo.

        :param video_path: ruta al archivo del vídeo.
        :return: stream del vídeo.
        """
        cap = cv2.VideoCapture(video_path)
        # Comprobar si el vídeo está disponible.
        if not cap.isOpened():
            raise Exception(f'The {video_path} can\'t be opened or doesn\'t exists.')
        return cap

    @property
    def fps(self) -> float:
        return self._fps

    @fps.setter
    def fps(self, value: float) -> None:
        self._fps = value

    @property
    def num_frames_available(self) -> int:
        """Número de frames total disponibles en la secuencia.

        :return: número de frames total disponible
        """
        return self._num_frames_available

    @num_frames_available.setter
    def num_frames_available(self, value: int) -> None:
        """Establecer el número de frames disponibles.

        Utilizar úniacmente para labores de depuración.

        :param value: cantidad de frames disponibles.
        :return:
        """
        self._num_frames_available = value

    @property
    def num_frames(self) -> int:
        """Cálculo del número de frames teniendo en cuenta el frame inicial y final establecido.

        :return: número de frames con los límites establecidos.
        """
        return self._end_frame - self._start_frame + 1


class StreamSequenceWriter:
    """Clase para la escritura de una secuencia de imágenes en un archivo de vídeo.
    """
    def release(self) -> None:
        """Cierra la conexión con el archivo.
        """
        # TODO

def load_sequence():
    raise DeprecationWarning('Use simple_object_detection.utils.video.StreamSequence class'
                             'instead.')


def save_sequence(sequence: StreamSequence, file_output: str) -> None:
    """Guarda una secuencia de frames como un vídeo.

    TODO: Eliminar el método y en su lugar crear una clase wrapper para VideoWriter e ir
     guardando el vídeo poco a poco.

    :param sequence: secuencia de frames.
    :param file_output: archivo donde se guardará (sobreescribe si ya existe).
    """
    # Cargar codec y video de salida.
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(file_output, fourcc, sequence.fps, (sequence.width, sequence.height))
    # Guardar cada frame de la secuencia.
    for frame in sequence:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    out.release()
