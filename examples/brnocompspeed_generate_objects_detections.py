"""Generate objects detections in BrnoCompSpeed sessions.
"""
import logging
import os

from simple_object_detection.models import YOLOv5s, YOLOv5m, YOLOv5l, YOLOv5x
from simple_object_detection.utils import save_objects_detections, generate_objects_detections
from simple_object_detection.utils.video import StreamSequence

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

session_folder = os.path.abspath(os.getenv('BRNOCOMPSPEED_FOLDER'))
cache_size = int(os.getenv('CACHE_SIZE', 100))
batch_size = int(os.getenv('BATCH_SIZE', 100))

sessions = [
    ('session1_center', os.path.join(session_folder, 'session1_center')),
    ('session1_left', os.path.join(session_folder, 'session1_left')),
    ('session1_right', os.path.join(session_folder, 'session1_right')),
    ('session2_center', os.path.join(session_folder, 'session2_center')),
    ('session2_left', os.path.join(session_folder, 'session2_left')),
    ('session2_right', os.path.join(session_folder, 'session2_right')),
    ('session3_center', os.path.join(session_folder, 'session3_center')),
    ('session3_left', os.path.join(session_folder, 'session3_left')),
    ('session3_right', os.path.join(session_folder, 'session3_right')),
    ('session4_center', os.path.join(session_folder, 'session4_center')),
    ('session4_left', os.path.join(session_folder, 'session4_left')),
    ('session4_right', os.path.join(session_folder, 'session4_right')),
    ('session5_center', os.path.join(session_folder, 'session5_center')),
    ('session5_left', os.path.join(session_folder, 'session5_left')),
    ('session5_right', os.path.join(session_folder, 'session5_right')),
    ('session6_center', os.path.join(session_folder, 'session6_center')),
    ('session6_left', os.path.join(session_folder, 'session6_left')),
    ('session6_right', os.path.join(session_folder, 'session6_right')),
]
networks = [YOLOv5s, YOLOv5m, YOLOv5l, YOLOv5x,]

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
        logger.info(f'Cargando el video {video_file} con un tamaño de buffer de {cache_size}')
        sequence = StreamSequence(video_file, cache_size=cache_size)
        logger.info(f'Generando detecciones en lotes de tamaño {batch_size}')
        objects_detections = generate_objects_detections(network, sequence, batch_size=batch_size,
                                                         verbose=True)
        save_objects_detections(objects_detections, output_file, pickle_version=4)
        logger.info('Detección de objetos terminada.')
