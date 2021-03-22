from simple_object_detection.utils import *

sequence = load_sequence('../../simple-object-tracking/sample_data/ball-8.mp4')
detections = load_detections_in_sequence(
    '../../simple-object-tracking/sample_data/detections/balls-8-detections-yolov3.pkl'
)

# Aplicar bounding boxes a cada frame.
for frame_id in range(len(sequence)):
    objects = detections[frame_id]
    sequence[frame_id] = set_bounding_boxes_in_image(sequence[frame_id], objects)

save_sequence(sequence, 1280, 720, 59.89, '../../output.mp4')
