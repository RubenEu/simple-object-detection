from typing import Tuple
from simple_object_detection.typing import Point2D


class Object:
    def __init__(self,
                 index: int,
                 center: Point2D,
                 width: int,
                 height: int,
                 score: float,
                 label: str,
                 **kwargs):
        """

        :param index: índice del objeto.
        :param center: centro del objeto.
        :param width: ancho de la caja que delimita al objeto.
        :param height: alto de la caja que delimita al objeto.
        :param score: puntuación.
        :param label: etiqueta del objeto.
        :param kwargs: otras propiedades e información del objeto.
        """
        self.index = index
        self._center = center
        self._bounding_box = self._create_bounding_box(center, width, height)
        self.width = width
        self.height = height
        self.score = score
        self.label = label
        self.other_properties = kwargs

    @property
    def center(self) -> Point2D:
        return self._center

    @center.setter
    def center(self, new_center: Point2D) -> None:
        self._center = new_center

    @property
    def bounding_box(self) -> Tuple[Point2D, Point2D, Point2D, Point2D]:
        """Devuelve los 4 puntos de la caja delimitadora en el orden de las agujas del reloj
        comenzando en la esquina superior izquierda.

        :return: puntos de las esquinas de la caja delimitadora.
        """
        return self._bounding_box

    @bounding_box.setter
    def bounding_box(self, new_bounding_box: Tuple[Point2D, Point2D, Point2D, Point2D]) -> None:
        self._bounding_box = new_bounding_box

    @staticmethod
    def _create_bounding_box(center, width, height):
        """Calcula los puntos de la caja delimitadora dado el centro, ancho y alto.

        La caja calculada es paralela a los ejes X e Y.

        :return: caja delimitadora.
        """
        center_x, center_y = center
        x_left, x_right = int(center_x - (width / 2)), int(center_x + (width / 2))
        y_top, y_bottom = int(center_y - (height / 2)), int(center_y + (height / 2))
        return (x_left, y_top), (x_right, y_top), (x_right, y_bottom), (x_left, y_bottom)

    def __str__(self):
        return f'ObjectDetected<center={self.center}, class={self.label}, score={self.score}>'

    def __repr__(self):
        return self.__str__()

