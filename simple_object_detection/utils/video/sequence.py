import cv2

from simple_object_detection.typing import Image


class Sequence:
    """Clase para cargar los frames de una secuencia de vídeo.

    Se utiliza la notación de acceso a un objeto ``object[item]``. Así se puede ir cargando
    el vídeo poco a poco sin llegar a saturar la memoria RAM.
    """
    def __init__(self, video_path: str):
        # Abrir el stream con OpenCV.
        self.stream = self._open_video_stream(video_path)
        # Información del vídeo.
        self.width: int = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height: int = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps: float = float(self.stream.get(cv2.CAP_PROP_FPS))
        self.num_frames: int = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))

    def __getitem__(self, item: int) -> Image:
        """
        TODO: Cachear. Cargar chunk. Etc. Etc.
        TODO: Añadir slicing:
        https://www.geeksforgeeks.org/implementing-slicing-in-__getitem__/#:~:text=slice%20is%20a%20constructor%20in,can%20be%20defined%20inside%20it.&text=Parameter%3A,constructor%20to%20create%20slice%20object.
        """
        # Comprobación del ítem.
        if not isinstance(item, int):
            raise TypeError()
        # Establecer en el frame que se busca.
        self.stream.set(cv2.CAP_PROP_POS_FRAMES, item)
        retval, frame_bgr = self.stream.retrieve()
        # Comprobar el valor de salida.
        if not retval:
            raise Exception('No se devolvió ningún frame.')
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return frame_rgb

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

