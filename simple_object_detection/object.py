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
        self._bounding_box = None
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
        """
        Devuelve los 4 puntos de la caja delimitadora en el orden de las agujas del reloj comenzando
        en la esquina superior izquierda.

        Si los puntos de la caja delimitadora no están almacenados en el atributo de instancia
        _bounding_box, los calcula a partir del centro y ancho (asume que la caja delimitadora es
        paralela a los ejes X e Y.

        :return: puntos de las esquinas de la caja delimitadora.
        """
        if self._bounding_box is None:
            center_x, center_y = self.center
            x_left, x_right = int(center_x - (self.width / 2)), int(center_x + (self.width / 2))
            y_top, y_bottom = int(center_y - (self.height / 2)), int(center_y + (self.height / 2))
            self._bounding_box = ((x_left, y_top), (x_right, y_top),
                                  (x_right, y_bottom), (x_left, y_bottom))
        return self._bounding_box

    @bounding_box.setter
    def bounding_box(self, new_bounding_box: Tuple[Point2D, Point2D, Point2D, Point2D]) -> None:
        self._bounding_box = new_bounding_box

    def __str__(self):
        return f'ObjectDetected<center={self.center}, class={self.label}, score={self.score}>'

    def __repr__(self):
        return self.__str__()

