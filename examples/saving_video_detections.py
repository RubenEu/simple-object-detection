"""Script para la creación de un archivo pickle con las detecciones en un vídeo.

Parámetros:
    1. Ruta al archivo de vídeo.
    2. Archivo de salida con las detecciones de objetos en el vídeo.
    3. Archivo de la máscara para aplicar.
    4. Nombre de la clase con la que realizar la detección.

TODO:
  - En un futuro, poner la máscara como opcional.
"""
import sys
from importlib import import_module

from simple_object_detection.utils import (load_image,
                                           save_objects_detections,
                                           generate_objects_detections)
from simple_object_detection.utils.video import StreamSequence

# Cargar vídeo.
video_path = sys.argv[1]
# Carpeta de salida.
file_output = sys.argv[2]
# Máscara para la detección de objetos.
mask_file = sys.argv[3]
mask = load_image(mask_file)
# Inicializar la red.
network_class_name = sys.argv[4].split('.')
network_class = getattr(import_module('.'.join(network_class_name[:-1])), network_class_name[-1])
network = network_class()
# Cargar la secuencia
sequence = StreamSequence(video_path, cache_size=200)
# Generar y guardar las detecciones.
object_detections = generate_objects_detections(network, sequence, mask)
save_objects_detections(object_detections, file_output)
