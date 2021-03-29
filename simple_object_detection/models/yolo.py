from ..detection_model import DetectionModel
from ..object import Object
import numpy as np
import cv2
import requests
import os
import torch


class YOLOv5s(DetectionModel):
    """
    YOLOv5s

    Realizado con:
        - https://github.com/ultralytics/yolov5
        - https://heartbeat.fritz.ai/a-2019-guide-to-object-detection-9509987954c3#3837
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Definir size del modelo (640, 640).
        self.model_size = 640

    def _load_local(self):
        raise NotImplemented('Model not implemeted for local use.')

    def _load_online(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        # Download and charge coco names
        r = requests.get(
            'https://raw.githubusercontent.com/amikelive/coco-labels/master/coco-labels-paper.txt')
        self.classes = [line.strip() for line in r.text.split('\n')]

    def _get_output(self, image):
        return self.model([image], size=self.model_size)

    def _calculate_number_detections(self, output, *args, **kwargs):
        return len(output.xyxy[0])

    def _calculate_bounding_box(self, output, obj_id, *args, **kwargs):
        xyxy = output.xyxy[0][obj_id]
        left, top, right, bottom = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
        return np.array(np.clip([left, right, top, bottom], 0, None), dtype=np.uint32)

    def _calculate_score(self, output, obj_id, *args, **kwargs):
        return float(output.xyxy[0][obj_id][4])

    def _calculate_label(self, output, obj_id, *args, **kwargs):
        class_id = int(output.xyxy[0][obj_id][5])
        return self.classes[class_id]


class YOLOv3(DetectionModel):
    """
    YOLOv3

    Explicación salidas YOLOv3 (416x416).
    -----------------------------------------
    Hay 3 neuronas en la capa final -> 3 salidas.
    Salida 1: ndarray (13x13x3, 85) (1/32 size)
    Salida 2: ndarray (26x26x3, 85) (1/16 size)
    Salida 3: ndarray (52x52x3, 85) (1/8 size)

    En el segundo eje se encuentran (4 + 1 + num_classes).
    Los 4 primeros elementos: center_x, center_y, width, height.
    El quinto elemento: ???
    Los 80 siguientes: la puntuación que recibe cada clase para el elemento detectado en la caja en ese punto.
    
    Uso local: se necesitan los archivos yolov3.cfg y yolov3.weights en la carpeta yolov3.
    Archivos necesarios:
        - https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
        - https://pjreddie.com/media/files/yolov3.weights
    Además, necesitará también un fichero de texto con el diccionario de clases de coco.
        - https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names

    Realizado con:
        - https://towardsdatascience.com/object-detection-using-yolov3-and-opencv-19ee0792a420
    """
    net = None
    classes = None
    layers_names = None
    output_layers = None

    def __init__(self, size=None, *args, **kwargs):
        """
        :param size: tamaño usado en la predicción. Opciones: ((320,320), (416,416), (608,608)). Por defecto: (320x320)
        """
        super().__init__(*args, **kwargs)
        # Parámetros para la predicción
        self.size = size if size else (320, 320)

    def _load_local(self):
        yolov3_weights = self.models_path + 'yolov3/yolov3.weights'
        yolov3_config = self.models_path + 'yolov3/yolov3.cfg'
        coco_names = self.models_path + 'coco.names'
        # Cargar modelo.
        self._load_dnn_network(yolov3_weights, yolov3_config, coco_names)

    def _load_online(self):
        self._download_models_files()
        yolov3_weights = self.temporal_folder + '/yolov3/yolov3.weights'
        yolov3_config = self.temporal_folder + '/yolov3/yolov3.cfg'
        coco_names = self.temporal_folder + '/yolov3/coco.names'
        # Cargar modelo.
        self._load_dnn_network(yolov3_weights, yolov3_config, coco_names)

    def _load_dnn_network(self, weights, config, coco):
        self.net = cv2.dnn.readNet(weights, config)
        self.classes = []
        with open(coco, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.layers_names = self.net.getLayerNames()
        self.output_layers = [self.layers_names[i[0] - 1] for i in
                              self.net.getUnconnectedOutLayers()]

    def _download_models_files(self):
        # TODO: Comprobar si están ya descargados.
        # Crear carpeta yolov3
        os.mkdir(self.temporal_folder + '/yolov3/')
        # Archivos para descargar.
        urls = [
            'https://pjreddie.com/media/files/yolov3.weights',
            'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg',
            'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'
        ]
        for url in urls:
            filename = url.split('/')[-1]
            r = requests.get(url)
            # https://stackoverflow.com/questions/16694907/download-large-file-in-python-with-requests
            with open(self.temporal_folder + '/yolov3/' + url.split('/')[-1], 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)

    def _get_output(self, image):
        blob = cv2.dnn.blobFromImage(image,
                                     scalefactor=0.00392,  # 1/255
                                     size=self.size,
                                     mean=(0, 0, 0),
                                     swapRB=True,
                                     crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        return outputs

    def _calculate_number_detections(self, output, *args, **kwargs):
        num_objects = 0
        for layer_out_id, layer_output in enumerate(output):
            for detect_id, detect in enumerate(layer_output):
                # Comprobar que ha obtenido una puntuación mínima.
                score = self._calculate_score(output, (layer_out_id, detect_id))
                if score > 0:
                    num_objects += 1
        return num_objects

    def _calculate_bounding_box(self, output, obj_id, *args, **kwargs):
        detect = output[obj_id[0]][obj_id[1]]
        output_box = detect[0:4]
        image = kwargs['image']
        height, width = image.shape[0:2]
        center_x, center_y = output_box[0] * width, output_box[1] * height
        box_width, box_height = output_box[2] * width, output_box[3] * height
        left, right = center_x - box_width / 2, center_x + box_width / 2,
        top, bottom = center_y - box_height / 2, center_y + box_height / 2
        bounding_box = np.array(np.clip([left, right, top, bottom], 0, None), dtype=np.uint32)
        return bounding_box

    def _calculate_score(self, output, obj_id, *args, **kwargs):
        detect = output[obj_id[0]][obj_id[1]]
        output_scores = detect[5:]
        return output_scores[np.argmax(output_scores)]

    def _calculate_label(self, output, obj_id, *args, **kwargs):
        detect = output[obj_id[0]][obj_id[1]]
        output_scores = detect[5:]
        class_idx = int(np.argmax(output_scores))
        return self.classes[class_idx]

    def _get_objects(self, image, output):
        num_detections = self._calculate_number_detections(output)
        objects = np.empty((num_detections,), dtype=Object)
        obj_id = 0
        # Añadir los objetos a la lista.
        for layer_out_id, layer_output in enumerate(output):
            for detect_id, detect in enumerate(layer_output):
                # Comprobar que ha obtenido una puntuación mínima.
                score = self._calculate_score(output, (layer_out_id, detect_id))
                if score > 0:
                    object_detected = Object(
                        index=obj_id,
                        bounding_box=self._calculate_bounding_box(output, (layer_out_id, detect_id), image=image),
                        score=score,
                        label=self._calculate_label(output, (layer_out_id, detect_id))
                    )
                    objects[obj_id] = object_detected
                    # Siguiente objeto.
                    obj_id += 1
        return objects
