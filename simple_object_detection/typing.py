import numpy as np

from typing import Type, Any, TypeVar, List, Tuple


Model: Type = Any
ModelOutput: Type = Any
Image: Type = TypeVar('Image',
                      Tuple[np.uint8, np.uint8, np.uint8],
                      np.ndarray)
Point2D: Type[tuple] = Tuple[int, int]
