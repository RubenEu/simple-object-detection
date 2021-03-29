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
        self.center = center
        self.width = width
        self.height = height
        self.score = score
        self.label = label
        self.other_properties = kwargs

    @property
    def bounding_box(self) -> Tuple[Point2D, Point2D, Point2D, Point2D]:
        """
        Devuelve los 4 puntos de la caja delimitadora en el orden de las agujas del reloj comenzando
        en la esquina superior izquierda.

        :return: puntos de las esquinas de la caja delimitadora.
        """
        center_x, center_y = self.center
        x_left, x_right = int(center_x - (self.width / 2)), int(center_x + (self.width / 2))
        y_top, y_bottom = int(center_y - (self.height / 2)), int(center_y + (self.height / 2))
        return (x_left, y_top), (x_right, y_top), (x_right, y_bottom), (x_left, y_bottom)

    def __str__(self):
        return f'ObjectDetected<id={self.index}, class={self.label}, score={self.score}>'

    def __repr__(self):
        return self.__str__()

