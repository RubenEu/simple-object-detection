import torch
from typing import Any

from simple_object_detection.detection_model import PyTorchHubModel
from simple_object_detection.typing import Image


class YOLOv5s(PyTorchHubModel):

    def _load_local(self) -> Any:
        raise NotImplementedError('YOLOv5s is not implemented for use with local files.')

    def _load_online(self) -> Any:
        return torch.hub.load('ultralytics/yolov5', 'yolov5s')

    def _get_output(self, image: Image) -> Any:
        return self.model([image], size=640)


class YOLOv5m(PyTorchHubModel):

    def _load_local(self) -> Any:
        raise NotImplementedError('YOLOv5m is not implemented for use with local files.')

    def _load_online(self) -> Any:
        return torch.hub.load('ultralytics/yolov5', 'yolov5m')

    def _get_output(self, image: Image) -> Any:
        return self.model([image], size=640)


class YOLOv5l(PyTorchHubModel):

    def _load_local(self) -> Any:
        raise NotImplementedError('YOLOv5l is not implemented for use with local files.')

    def _load_online(self) -> Any:
        return torch.hub.load('ultralytics/yolov5', 'yolov5l')

    def _get_output(self, image: Image) -> Any:
        return self.model([image], size=640)


class YOLOv5x(PyTorchHubModel):

    def _load_local(self) -> Any:
        raise NotImplementedError('YOLOv5s is not implemented for use with local files.')

    def _load_online(self) -> Any:
        return torch.hub.load('ultralytics/yolov5', 'yolov5x')

    def _get_output(self, image: Image) -> Any:
        return self.model([image], size=640)


class YOLOv5s6(PyTorchHubModel):

    def _load_local(self) -> Any:
        raise NotImplementedError('YOLOv5s is not implemented for use with local files.')

    def _load_online(self) -> Any:
        return torch.hub.load('ultralytics/yolov5', 'yolov5s6')

    def _get_output(self, image: Image) -> Any:
        return self.model([image], size=1280)


class YOLOv5m6(PyTorchHubModel):

    def _load_local(self) -> Any:
        raise NotImplementedError('YOLOv5s is not implemented for use with local files.')

    def _load_online(self) -> Any:
        return torch.hub.load('ultralytics/yolov5', 'yolov5m6')

    def _get_output(self, image: Image) -> Any:
        return self.model([image], size=1280)


class YOLOv5l6(PyTorchHubModel):

    def _load_local(self) -> Any:
        raise NotImplementedError('YOLOv5s is not implemented for use with local files.')

    def _load_online(self) -> Any:
        return torch.hub.load('ultralytics/yolov5', 'yolov5l6')

    def _get_output(self, image: Image) -> Any:
        return self.model([image], size=1280)


class YOLOv5x6(PyTorchHubModel):

    def _load_local(self) -> Any:
        raise NotImplementedError('YOLOv5s is not implemented for use with local files.')

    def _load_online(self) -> Any:
        return torch.hub.load('ultralytics/yolov5', 'yolov5x6')

    def _get_output(self, image: Image) -> Any:
        return self.model([image], size=1280)
