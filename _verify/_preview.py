"""Draw final placed elements onto each page for an inline visual preview.
Reuses the grounding cache (no API calls). Not part of the pipeline."""
import sys
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import pipeline

CACHE = Path("/tmp/img2form_test")
COL = {"text": (194, 113, 25), "textarea": (194, 113, 25), "number": (153, 133, 12),
       "email": (194, 113, 25), "phone": (194, 113, 25), "date": (181, 54, 156),
       "time": (181, 54, 156), "radio": (12, 89, 232), "checkbox_group": (12, 89, 232),
       "select": (12, 89, 232), "consent": (62, 138, 43), "signature": (68, 158, 47),
       "image": (217, 65, 103)}

for idx, out in [(23, "overlay_p09.png"), (24, "overlay_p10.png")]:
    name, path = pipeline_docs = __import__("render").DOCS[idx]
    built = pipeline.build(path, cache_path=str(CACHE / f"gcache_{idx}.json"))
    im = built["img"].copy()
    for it in built["elements"]:
        x, y, w, h = it["rect"]
        c = COL.get(it["type"], (120, 120, 120))
        cv2.rectangle(im, (x, y), (x + w, y + h), c, 2)
    dst = ROOT / "_verify" / "아동정서발달" / out
    cv2.imwrite(str(dst), im)
    print(name, len(built["elements"]), "->", dst)
