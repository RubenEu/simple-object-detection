import tensorflow_hub as hub
from .tfhub_models import TFHubModel


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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        local_module_handle = 'faster_rcnn_openimages_v4_inception_resnet_v2_1'
        module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
        # Cargar el modelo localmente.
        if self.offline_mode:
            self.detector = hub.load(self.models_path + local_module_handle).signatures['default']
        # Descargar el modelo online.
        if not self.offline_mode:
            self.detector = hub.load(module_handle).signatures['default']

