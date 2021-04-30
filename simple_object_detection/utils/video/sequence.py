from typing import List, Tuple, Union, Optional

import cv2

from simple_object_detection.typing import Image


class Sequence:
    """Clase para cargar los frames de una secuencia de vídeo.

    Se utiliza la notación de acceso a un objeto ``object[item]``. Así se puede ir cargando
    el vídeo poco a poco sin llegar a saturar la memoria RAM.
    """
    def __init__(self, video_path: str, cache_size: int = 100):
        # Abrir el stream con OpenCV.
        self.stream = self._open_video_stream(video_path)
        # Información del vídeo.
        self.width: int = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height: int = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps: float = float(self.stream.get(cv2.CAP_PROP_FPS))
        self.num_frames: int = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
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

    def _get_frame(self, fid: int) -> Image:
        """

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
        """

        :param id:
        :return:
        """
        expected_index = fid % len(self._cache)
        cached_item: Tuple[int, Image] = self._cache[expected_index]
        frame_id, frame = self._cache[expected_index]
        if frame_id == fid:
            return frame
        return None

    def _pull_to_cache(self, fid) -> Image:
        """Trae a caché los siguientes frames y devuelve el buscado.

        :return:
        """
        # Establecer en el frame que se busca.
        retval = self.stream.set(cv2.CAP_PROP_POS_FRAMES, fid)
        if not retval:
            raise Exception('Ocurrió un error al posicionar el número de frame.')
        # Añadir los siguientes frames que quepan la caché.
        num_frames_pulled = 0
        while True:
            # Capturar frame a frame.
            actual_frame_id = int(self.stream.get(cv2.CAP_PROP_POS_FRAMES))
            ret, frame = self.stream.read()
            # Comprobar si se ha leído el frame correctamente.
            if not ret:
                # TODO: Aquí a veces lanza error de que no se ha recibido.
                #  Cuando sí debería recibirse.
                print(f"No se ha recibido el frame {actual_frame_id} (¿stream finalizado?)."
                      f"Saliendo...")
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

