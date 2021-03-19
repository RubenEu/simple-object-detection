import requests
import tensorflow as tf
import tensorflow_hub as hub
from .tfhub_models import TFHubModel


class CenterNetHourGlass(TFHubModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Cargar nombres de clases
        # Offline.
        if self.offline_mode:
            coco_names = self.models_path + 'coco.names'
        # Online (descargar archivos).
        if not self.offline_mode:
            r = requests.get('https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names')
            open(self.temporal_folder + '/' + 'coco.names', 'wb').write(r.content)
            coco_names = self.temporal_folder + '/yolov3/coco.names'
        self.classes = []
        with open(coco_names, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

    def _preprocess_image(self, image):
        return tf.image.convert_image_dtype(image, tf.uint8)[tf.newaxis, ...]

    def _get_objects(self, image, output):
        # Por alguan razón, en la salida de cada clave del diccionario, tiene un primer 'eje' con 1 solo elemento,
        # por tanto, se elimina y se delega en la clase padre.
        output_squeezed = dict()
        for key, value in output.items():
            output_squeezed[key] = tf.squeeze(output[key], axis=0)
        # Devuelve el resultado de ejecutar el método del padre con el output preprocesado.
        return super()._get_objects(image, output_squeezed)

    def _calculate_label(self, output, obj_id, *args, **kwargs):
        # No devuelve la palabra en la propia salida, utiliza el diccionario de palabras de COCO.
        output_class_id = int(output['detection_classes'][obj_id])
        return self.classes[output_class_id]


class CenterNetHourGlass104512x512(CenterNetHourGlass):
    """
    CenterNet Object detection model with the Hourglass backbone,
    trained on COCO 2017 dataset with trainning images scaled to 512x512.

    Uso local: se tendrá que descargar el archivo .tar.gz en una carpeta con el mismo nombre que él,
        el archivo se puede encontrar en:
        https://tfhub.dev/tensorflow/centernet/hourglass_512x512/1?tf-hub-format=compressed
    
    Realizado con:
        - https://tfhub.dev/tensorflow/centernet/hourglass_512x512/1
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        local_module_handle = 'centernet_hourglass_512x512_1'
        module_handle = "https://tfhub.dev/tensorflow/centernet/hourglass_512x512/1"
        # Cargar el modelo localmente.
        if self.offline_mode:
            self.detector = hub.load(self.models_path + local_module_handle)
        # Descargar el modelo online.
        if not self.offline_mode:
            self.detector = hub.load(module_handle)


class CenterNetHourGlass1041024x1024(CenterNetHourGlass):
    """
    CenterNet Object detection model with the Hourglass backbone,
    trained on COCO 2017 dataset with trainning images scaled to 1024x1024.

    Uso local: se tendrá que descargar el archivo .tar.gz en una carpeta con el mismo nombre que él,
        el archivo se puede encontrar en:
        https://tfhub.dev/tensorflow/centernet/hourglass_1024x1024/1?tf-hub-format=compressed

    Realizado con:
        - https://tfhub.dev/tensorflow/centernet/hourglass_1024x1024/1
        - https://www.tensorflow.org/hub/tutorials/tf2_object_detection?hl=hu
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        local_module_handle = 'centernet_hourglass_1024x1024_1'
        module_handle = "https://tfhub.dev/tensorflow/centernet/hourglass_1024x1024/1"
        # Cargar el modelo localmente.
        if self.offline_mode:
            self.detector = hub.load(self.models_path + local_module_handle)
        # Descargar el modelo online.
        if not self.offline_mode:
            self.detector = hub.load(module_handle)
