import numpy as np

from typing import List, Type, Any, TypeVar, Tuple, NamedTuple

Model: Type = Any
ModelOutput: Type = Any
Image: Type = TypeVar('Image',
                      Tuple[np.uint8, np.uint8, np.uint8],
                      np.ndarray)


class Point2D(NamedTuple):
    """Representa un punto 2D."""
    x: int
    y: int


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
