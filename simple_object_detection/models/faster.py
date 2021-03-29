import tensorflow_hub as hub

from simple_object_detection.detection_model import TFHubModel


class FasterRCNNInceptionResnetV2(TFHubModel):
    """
    FasterRCNN+InceptionResNetV2 network trained on Open Images V4.

    Uso local: se tendrá que descargar el archivo .tar.gz en una carpeta con el mismo nombre que él,
        el archivo se puede encontrar en:
        https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1?tf-hub-format=compressed

    Realizado con:
        - https://www.tensorflow.org/hub/tutorials/object_detection
        - https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1
    """

    def _load_local(self):
        local_module_handle = 'faster_rcnn_openimages_v4_inception_resnet_v2_1'
        return hub.load(self.models_path + local_module_handle).signatures['default']

    def _load_online(self):
        module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
        return hub.load(module_handle).signatures['default']

