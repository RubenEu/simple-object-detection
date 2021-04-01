import numpy as np

from typing import List, Type, Any, TypeVar, Tuple


Model: Type = Any
ModelOutput: Type = Any
Image: Type = TypeVar('Image',
                      Tuple[np.uint8, np.uint8, np.uint8],
                      np.ndarray)
Point2D: Type[tuple] = Tuple[int, int]
# Secuencia con información (width, height, fps, frames, timestamps).
SequenceLoaded = Tuple[int, int, float, List[Image], List[int]]
