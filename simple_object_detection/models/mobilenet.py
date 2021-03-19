import tensorflow_hub as hub
from .tfhub_models import TFHubModel


class SSDMobileNetV2(TFHubModel):
    """
    SSD+MobileNetV2

    Uso local: se tendrá que descargar el archivo .tar.gz en una carpeta con el mismo nombre que él,
        el archivo se puede encontrar en:
        https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1?tf-hub-format=compressed
        
    Realizado con:
        - https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1
    """

    def _load_online(self):
        super()._load_online()
        module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
        self.detector = hub.load(module_handle).signatures['default']

    def _load_local(self):
        super()._load_local()
        local_module_handle = 'openimages_v4_ssd_mobilenet_v2_1'
        self.detector = hub.load(self.models_path + local_module_handle).signatures['default']

