from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.val import YOLOEDetectValidator

from vista.models.validator import VISTAOutputMixin


class YOLOEValidator(VISTAOutputMixin, YOLOEDetectValidator):
    """YOLOEDetectValidator that also writes VISTA structured JSON/CSV outputs."""


class YOLOEVista(YOLOE):
    def __init__(self, *args, names: list[str] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fuse()
        if names is not None:
            self.set_classes(names)

    @property
    def task_map(self):
        tm = super().task_map
        tm["detect"]["validator"] = YOLOEValidator
        return tm
