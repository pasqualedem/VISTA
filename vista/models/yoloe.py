from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.val import YOLOEDetectValidator

from vista.models.validator import VISTAOutputMixin


class YOLOEValidator(VISTAOutputMixin, YOLOEDetectValidator):
    """YOLOEDetectValidator that also writes VISTA structured JSON/CSV outputs."""


class YOLOESegAsDetectValidator(YOLOEValidator):
    """Validate a seg model on detection-only labels.

    Seg models return (det_preds, proto) tuples; DetectionValidator.postprocess
    expects a plain tensor, so we unwrap here before delegating.
    """

    def postprocess(self, preds):
        if isinstance(preds, (list, tuple)):
            preds = (preds [0][0], preds[1])
        return super().postprocess(preds)
    

class YOLOEVista(YOLOE):
    def __init__(self, *args, names: list[str] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if names is not None:
            self.set_classes(names)
            self.fuse()

    @property
    def task_map(self):
        tm = super().task_map
        tm["detect"]["validator"] = YOLOEValidator
        tm["segment"]["validator"] = YOLOESegAsDetectValidator
        return tm

    def val(self, *args, **kwargs):
        # Force detect task so the detect validator and class-remapping pipeline are
        # used regardless of whether the loaded weights are a seg checkpoint.
        return super().val(*args, task="detect", **kwargs)
