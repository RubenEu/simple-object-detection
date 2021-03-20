# Simple object detection

Librería para la extracción de características de la detección de objetos con diferentes
modelos de redes neuronales.

## Características de la librería

Los modelos podrán ser descargados a la hora de ser cargados (instanciándolos) o se podrán cargar
desde archivos locales, para ello hay que configurar el directorio y establecer los archivos del
modelo en la carpeta correspondiente.

### Cargar modelo de forma online.

Esta es la opción por defecto, descargar a una carpeta temporal los archivos necesarios para
instanciar el modelo.

```python
from simple_object_detection.models import YOLOv3
network = YOLOv3(size=(608, 608))
```

### Cargar modelo usando archivos locales.

Primeramente hay que configurar la ruta a la carpeta donde se almacenan los modelos, y después
descargar los archivos necesarios para su utilización.

Crearemos la carpeta models_data/yolov3 y en ella deben estar los siguientes archivos:
- [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)
- [yolov3.cfg](https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg)
- [coco.names](https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names)

```python
from simple_object_detection import DetectionModel
from simple_object_detection.models import YOLOv3

DetectionModel.models_path = 'models_data/'
network = YOLOv3(use_local=True)
```

Las clases de los modelos deben implementar la clase abstracta [DetecionModel](detection_model.py).

Los métodos a sobrescribir son:

- **_load_local(self)**: carga el modelo desde la carpeta de modelos local.

- **_load_online(self)**: carga el modelo descargándolo previamente.
  
- **_get_output(self, image)**: tendrá como entrada la imagen sin preprocesar. Debe devolver la salida de la red
neuronal. El formato de salida será el que cada red utilice.
   
- **_calculate_number_detections(self, output, \*args, \*\*kwargs)**: este método será llamado para obtener el número
de detecciones que ha realizado la red. Su salida será un número entero.

- **_calculate_bounding_box(self, output, obj_id, \*args, \*\*kwargs)**: recibe como parámetro la salida y la posición
del objeto. La salida debe ser del siguiente formato: ndarray 4 posiciones de tipo uint32 donde se indiquen
4 enteros indicando la posición [left, right, top, bottom], o lo que es lo mismo, la posición mínima de la
izquierda, la posición máxima de la derecha, la posición más alta y la más baja de la caja delimitadora.

- **_calculate_score(self, output, obj_id, \*args, \*\*kwargs)**: recibe la salida de la red y el id del objeto y debe
devolver la puntuación que ha obtenido ese objeto.

- **_calculate_label(self, output, obj_id, \*args, \*\*kwargs)**: recibe la salida de la red y el id del objeto y debe
devolver el string de la etiqueta de la clase a la que pertenece el objeto.

### Ejemplo de clase con los métodos necesarios para sobreescribir.

```python
from simple_object_detection import DetectionModel

class MyExampleModel(DetectionModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _load_local(self):
        return None
    
    def _load_online(self):
        return None
    
    def _get_output(self, image):
        return None

    def _get_objects(self, image, output):
        return None

    def _calculate_number_detections(self, output, *args, **kwargs):
        return None

    def _calculate_bounding_box(self, output, obj_id, *args, **kwargs):
        return None

    def _calculate_score(self, output, obj_id, *args, **kwargs):
        return None

    def _calculate_label(self, output, obj_id, *args, **kwargs):
        return None

```

## Modelos implementados

#### YOLOv3

Es el modelo más rápido de los actualmente implementados.  
Se puede instanciar con el parámetro opcional size con una tupla de las dimensiones de
imagen entre las siguientes disponibles: (320,320), (416,416), (608,608). Por defecto
se utiliza (320, 320).

```python
from simple_object_detection.models import YOLOv3

network = YOLOv3(size=(608, 608))
```

Para utilizar localmente hace falta crear la carpeta **yolov3** e introducir los
siguientes archivos:
- [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)
- [yolov3.cfg](https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg)
- [coco.names](https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names)

### Modelos de tensorflow-hub

Para el uso local de cualquiera de ellos basta con descargar el archivo .tar.gz desde
tensorflow-hub y descomprimirlo en una carpeta con el mismo nombre del archivo.

#### SSD+MobileNetV2

```python
from simple_object_detection.models import SSDMobileNetV2

network = SSDMobileNetV2()
```

#### CenterNet HourGlass104 512x512

```python
from simple_object_detection.models import CenterNetHourGlass104512x512

network = CenterNetHourGlass104512x512()
```

Para el uso de este modelo de forma localse necesita además el archivo de nombres
[coco](https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names).

#### CenterNet HourGlass104 1024x1024

```python
from simple_object_detection.models import CenterNetHourGlass1041024x1024

network = CenterNetHourGlass1041024x1024()
```

Para el uso de este modelo de forma localse necesita además el archivo de nombres
[coco](https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names).

#### FasterRCNN+InceptionResNetV2

```python
from simple_object_detection.models import FasterRCNNInceptionResnetV2

network = FasterRCNNInceptionResnetV2()
```
