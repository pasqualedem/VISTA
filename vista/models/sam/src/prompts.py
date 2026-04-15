"""
Text prompts associated with each dataset class id.
"""

from __future__ import annotations

from typing import Dict

# Class mapping from the YOLO dataset:
# 0: crashed car
# 1: person
# 2: car

CLASS_PROMPTS: Dict[int, str] = {
    0: "crashed car",
    1: "person",
    2: "car",
}
