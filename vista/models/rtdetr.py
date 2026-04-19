from ultralytics import RTDETR
from ultralytics.models.rtdetr.val import RTDETRValidator as _RTDETRValidator

from vista.models.validator import VISTAOutputMixin


class RTDETRValidator(VISTAOutputMixin, _RTDETRValidator):
    """RTDETRDetectValidator that also writes VISTA structured JSON/CSV outputs."""


class RTDETRVista(RTDETR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def task_map(self):
        tm = super().task_map
        tm["detect"]["validator"] = RTDETRValidator
        return tm
