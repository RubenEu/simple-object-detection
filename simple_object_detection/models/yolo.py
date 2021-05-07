import torch
from typing import Any

from simple_object_detection.detection_model import PyTorchHubModel


class YOLOv5s(PyTorchHubModel):
    size = 640

    def _load_local(self) -> Any:
        raise NotImplementedError('YOLOv5s is not implemented for use with local files.')

    def _load_online(self) -> Any:
        return torch.hub.load('ultralytics/yolov5', 'yolov5s')


class YOLOv5m(PyTorchHubModel):
    size = 640

    def _load_local(self) -> Any:
        raise NotImplementedError('YOLOv5m is not implemented for use with local files.')

    def _load_online(self) -> Any:
        return torch.hub.load('ultralytics/yolov5', 'yolov5m')


class YOLOv5l(PyTorchHubModel):
    size = 640

    def _load_local(self) -> Any:
        raise NotImplementedError('YOLOv5l is not implemented for use with local files.')

    def _load_online(self) -> Any:
        return torch.hub.load('ultralytics/yolov5', 'yolov5l')


class YOLOv5x(PyTorchHubModel):
    size = 640

    def _load_local(self) -> Any:
        raise NotImplementedError('YOLOv5s is not implemented for use with local files.')

    def _load_online(self) -> Any:
        return torch.hub.load('ultralytics/yolov5', 'yolov5x')


class YOLOv5s6(PyTorchHubModel):
    size = 1280

    def _load_local(self) -> Any:
        raise NotImplementedError('YOLOv5s is not implemented for use with local files.')

    def _load_online(self) -> Any:
        return torch.hub.load('ultralytics/yolov5', 'yolov5s6')


class YOLOv5m6(PyTorchHubModel):
    size = 1280

    def _load_local(self) -> Any:
        raise NotImplementedError('YOLOv5s is not implemented for use with local files.')

    def _load_online(self) -> Any:
        return torch.hub.load('ultralytics/yolov5', 'yolov5m6')


class YOLOv5l6(PyTorchHubModel):
    size = 1280

    def _load_local(self) -> Any:
        raise NotImplementedError('YOLOv5s is not implemented for use with local files.')

    def _load_online(self) -> Any:
        return torch.hub.load('ultralytics/yolov5', 'yolov5l6')


class YOLOv5x6(PyTorchHubModel):
    size = 1280

    def _load_local(self) -> Any:
        raise NotImplementedError('YOLOv5s is not implemented for use with local files.')

    def _load_online(self) -> Any:
        return torch.hub.load('ultralytics/yolov5', 'yolov5x6')
