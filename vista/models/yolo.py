from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator

from vista.models.validator import VISTAOutputMixin


class YOLOValidator(VISTAOutputMixin, DetectionValidator):
    """YOLODetectValidator that also writes VISTA structured JSON/CSV outputs."""


class YOLOVista(YOLO):
    def __init__(self, *args, names: list[str] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fuse()
        if names is not None:
            self.set_classes(names)

    @property
    def task_map(self):
        tm = super().task_map
        tm["detect"]["validator"] = YOLOValidator
        return tm
