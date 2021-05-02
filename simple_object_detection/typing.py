import numpy as np

from typing import Union, Tuple, NamedTuple


Image = Union[Tuple[np.uint8, np.uint8, np.uint8], np.ndarray]


class Point2D(NamedTuple):
    """Representa un punto 2D."""
    x: int
    y: int


class FloatVector2D(NamedTuple):
    """Representa un vector en 2D de tipo float."""
    x: float
    y: float


class BoundingBox(NamedTuple):
    """Representa una caja delimitadora."""
    top_left: Point2D
    top_right: Point2D
    bottom_right: Point2D
    bottom_left: Point2D


class RelativeBoundingBox(NamedTuple):
    """Representa una caja delimitadora dado su centro y su alto y ancho."""
    center: Point2D
    width: int
    height: int


class VideoProperties(NamedTuple):
    width: int
    height: int
    fps: float
    num_frames: int
