from typing import List, Tuple, Union, Optional

import cv2

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
        self._num_frames: int = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        # Caching (Almacena el número del frame y la imagen).
        self._cache: List[Tuple[int, Image]] = [(..., ...)] * cache_size

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
        # Extraer el frame buscado.
        frame_bgr = self._get_frame(item)
        # Comprobar el valor de salida.
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return frame_rgb

    def __len__(self):
        return self.num_frames

    def _get_frame(self, fid: int) -> Image:
        """TODO: Documentar.

        :param id:
        :return:
        """
        # Comprobación del índice.
        if fid >= self.num_frames:
            raise IndexError('Se ha excedido el límite.')
        # Buscar en caché primero.
        cached = self._search_in_cache(fid)
        if cached is None:
            return self._pull_to_cache(fid)
        return cached

    def _search_in_cache(self, fid: int) -> Optional[Image]:
        """TODO: Documentar.

        :param id:
        :return:
        """
        expected_index = fid % len(self._cache)
        cached_item: Tuple[int, Image] = self._cache[expected_index]
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
        while actual_frame_id < self.num_frames:
            # Capturar frame a frame.
            actual_frame_id = int(self.stream.get(cv2.CAP_PROP_POS_FRAMES))
            ret, frame = self.stream.read()
            # Comprobar si se ha leído el frame correctamente.
            if not ret:
                # TODO: Aquí a veces lanza error de que no se ha recibido.
                #  Cuando sí debería recibirse.
                # TODO: Ver cuántos fps tiene el vídeo y cuántos se leen aquí.
                #  igual eso es todo el error.
                #  Además, manejar si hay 153 frames y se está buscando el 154, que ya no se pid
                #  más.
                # o que ese sea el final...
                break
            # Añadir a la caché
            self._cache[actual_frame_id % len(self._cache)] = (actual_frame_id, frame)
            # Comprobar si se ha rellenado la caché.
            if (actual_frame_id + 1) % len(self._cache) == 0:
                break
        # Devolver el frame que se buscaba.
        return self._cache[fid % len(self._cache)][1]

    def __del__(self):
        """Libera el recurso del vídeo cargado y elimina el objeto también.
        """
        self.stream.release()
        del self.stream

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
    def fps(self):
        return self._fps

    @fps.setter
    def fps(self, value: float):
        self._fps = value

    @property
    def num_frames(self):
        return self._num_frames

    @num_frames.setter
    def num_frames(self, value: int):
        self._num_frames = value


def load_sequence():
    raise DeprecationWarning('Use simple_object_detection.utils.video.StreamSequence class instead.')


def save_sequence(sequence: StreamSequence, file_output: str) -> None:
    """Guarda una secuencia de frames como un vídeo.

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
