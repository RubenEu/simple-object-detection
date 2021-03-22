from simple_object_detection import DetectionModel
from simple_object_detection.models import *
from simple_object_detection.utils import *

_, _, _, sequence = load_sequence('../../simple-object-tracking/sample_data/ball-10.mp4')
# Configurar la ruta de los modelos (local).
DetectionModel.models_path = '../models_data/'

# Configurar la red y guardar detecciones.
net = YOLOv3(use_local=True)
save_detections_in_sequence(net, sequence, '../../balls-10-detections-yolov3.pkl')

# Para todos los videos:
# # Configurar la ruta de los modelos (local).
# DetectionModel.models_path = '../models_data/'
# net = FasterRCNNInceptionResnetV2(use_local=True)
# for i in range(9):
#     sequence = load_sequence('../../simple-object-tracking/sample_data/ball-' + str(i) + '.mp4')
#     save_detections_in_sequence(net, sequence, '../../balls-' + str(i) + '-detections.pkl')
