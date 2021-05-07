import logging
import os

from simple_object_detection.models import YOLOv5s, YOLOv5m, YOLOv5l, YOLOv5x
from simple_object_detection.utils import save_objects_detections, generate_objects_detections
from simple_object_detection.utils.video import StreamSequence

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

session_folder = os.path.abspath(os.getenv('BRNOCOMPSPEED_FOLDER'))

sessions = [
    ('session1_center', os.path.join(session_folder, 'session1_center')),
    ('session1_left', os.path.join(session_folder, 'session1_left')),
    ('session1_right', os.path.join(session_folder, 'session1_right')),
    ('session2_center', os.path.join(session_folder, 'session2_center')),
    ('session4_center', os.path.join(session_folder, 'session2_center')),
]
networks = [
    YOLOv5s,
    YOLOv5m,
    YOLOv5l,
    YOLOv5x,
]

for network_cls in networks:
    # Initialize network
    network = network_cls()
    for session in sessions:
        session_name, session_path = session
        # Variables.
        video_file = os.path.join(session_path, 'video.mp4')
        output_file = os.path.join(session_path, f'{network.__class__.__name__.lower()}.pkl')
        # Logging.
        logger.info(f'Generando las detecciones para {session_name}')
        logger.info(f'Salida en el archivo {output_file}')
        # Cargar secuencia y realizar detecciones.
        sequence = StreamSequence(video_file, cache_size=300)
        objects_detections = generate_objects_detections(network, sequence, batch_size=100)
        save_objects_detections(objects_detections, output_file, pickle_version=4)
        logger.info('Detección de objetos terminada.')
