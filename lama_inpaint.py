#!/usr/bin/env python
"""LaMa image inpainting through OpenCV's DNN module.

The ONNX export in ``models/lama.onnx`` accepts a fixed 512x512 image plus
binary mask and regenerates the masked region, so arbitrarily sized images
are handled by cropping a padded window around the mask, running the
network at 512x512, and pasting the result back at full resolution.  This
keeps detail that would be lost by squashing the whole image to 512x512.

Mask convention: pixels > 127 mark the region to inpaint, everything else
is kept.  Note this is the opposite of the scene-completion masks bundled
in ``sample_images``, where black marks the hole.

Standalone usage:
    uv run python lama_inpaint.py [image] [mask]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "lama.onnx"
OUTPUTS_DIR = REPO_ROOT / "outputs"

LAMA_MODEL_SIZE = 512  # fixed input resolution of the exported model

# padding around the mask bounding box, as a fraction of its larger side,
# clamped so tiny holes still get context and huge holes do not explode
PAD_MIN_PX = 32
PAD_MAX_PX = 256
PAD_FRACTION = 0.5

_NET = None
_NET_PATH: Optional[Path] = None


def _get_net(model_path=None):
    """Load the ONNX model once and reuse it across calls."""
    global _NET, _NET_PATH
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if _NET is None or _NET_PATH != path:
        if not path.exists():
            raise FileNotFoundError(f"LaMa model not found: {path}")
        t0 = time.time()
        _NET = cv2.dnn.readNetFromONNX(str(path), engine=cv2.dnn.ENGINE_AUTO)
        _NET_PATH = path
        print(f"  LaMa: loaded {path.name}  [{time.time() - t0:.2f}s]")
    return _NET


def _as_uint8_bgr(img: np.ndarray) -> np.ndarray:
    """Coerce an image (float or int, BGR) into clipped uint8."""
    return np.clip(img, 0, 255).astype(np.uint8)


def _as_single_channel(mask: np.ndarray) -> np.ndarray:
    """Collapse a possibly 3 channel mask to single channel."""
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    return mask


def _binary_mask(mask: np.ndarray) -> np.ndarray:
    """Binarise a mask to uint8 {0, 255} where 255 = region to inpaint."""
    mask = _as_single_channel(mask)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return binary


def _window_around(mask_bin: np.ndarray) -> Tuple[int, int, int, int]:
    """Padded pixel window (y0, y1, x0, x1) enclosing the masked pixels."""
    ys, xs = np.where(mask_bin > 0)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    pad = int(min(PAD_MAX_PX, max(PAD_MIN_PX, PAD_FRACTION * max(y1 - y0, x1 - x0))))

    h, w = mask_bin.shape[:2]
    wy0, wy1 = max(y0 - pad, 0), min(y1 + pad, h)
    wx0, wx1 = max(x0 - pad, 0), min(x1 + pad, w)
    return wy0, wy1, wx0, wx1


def _forward_512(net, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """One 512x512 LaMa pass on pre-resized uint8 BGR image and mask."""
    image_blob = cv2.dnn.blobFromImage(image, 1 / 255.0,
                                       (LAMA_MODEL_SIZE, LAMA_MODEL_SIZE))
    mask_blob = cv2.dnn.blobFromImage(mask, scalefactor=1.0,
                                      size=(LAMA_MODEL_SIZE, LAMA_MODEL_SIZE),
                                      mean=(0,), swapRB=False, crop=False)
    mask_blob = (mask_blob > 0).astype(np.float32)

    net.setInput(image_blob, "image")
    net.setInput(mask_blob, "mask")
    output = net.forward()

    result = output[0].transpose(1, 2, 0)
    return np.clip(result, 0, 255)


def lama_inpaint(img: np.ndarray, mask: np.ndarray,
                 model_path=None) -> Optional[np.ndarray]:
    """Inpaint the ``mask`` region (pixels > 127) of BGR ``img`` with LaMa.

    A padded window around the mask is resampled to the network's fixed
    512x512 input, inpainted, resampled back and written into a full
    resolution copy of ``img``.  Pixels far from the mask are returned
    untouched.

    Returns the inpainted uint8 BGR image, or ``None`` if the model file
    is missing.
    """
    src = _as_uint8_bgr(img)
    mask_bin = _binary_mask(mask)

    if mask_bin.shape[:2] != src.shape[:2]:
        raise ValueError(
            f"image {src.shape[:2]} and mask {mask_bin.shape[:2]} differ in size")
    if not (mask_bin > 0).any():
        return src.copy()

    try:
        net = _get_net(model_path)
    except FileNotFoundError as exc:
        print(f"  [!] {exc}")
        return None

    wy0, wy1, wx0, wx1 = _window_around(mask_bin)
    crop = src[wy0:wy1, wx0:wx1]
    crop_mask = mask_bin[wy0:wy1, wx0:wx1]
    ch, cw = crop.shape[:2]

    t0 = time.time()
    interp = cv2.INTER_AREA if max(ch, cw) > LAMA_MODEL_SIZE else cv2.INTER_LINEAR
    resized_img = cv2.resize(crop, (LAMA_MODEL_SIZE, LAMA_MODEL_SIZE),
                             interpolation=interp)
    resized_mask = cv2.resize(crop_mask, (LAMA_MODEL_SIZE, LAMA_MODEL_SIZE),
                              interpolation=cv2.INTER_NEAREST)

    result = _forward_512(net, resized_img, resized_mask)
    result = cv2.resize(result, (cw, ch), interpolation=cv2.INTER_LINEAR)
    result = _as_uint8_bgr(result)

    out = src.copy()
    out[wy0:wy1, wx0:wx1] = result
    print(f"  LaMa: inpainted {cw}x{ch} px window "
          f"[{time.time() - t0:.2f}s]")
    return out


# ---------------------------------------------------------------------------
# standalone demo
# ---------------------------------------------------------------------------

def run_inpaint(image_path, mask_path) -> None:
    """CLI demo: inpaint ``image_path`` where ``mask_path`` is white."""
    print("=" * 60)
    print("  LaMa Inpainting Demo")
    print("=" * 60)

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  Could not read image: {image_path}")
        return
    h, w = img.shape[:2]
    print(f"  Image: {w}x{h}")

    mask = cv2.imread(str(mask_path))
    if mask is None:
        print(f"  Could not read mask: {mask_path}")
        return
    mh, mw = mask.shape[:2]
    print(f"  Mask:  {mw}x{mh}")

    if (h, w) != (mh, mw):
        print("  [!] Image and mask dimensions mismatch! Resizing mask.")
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = _binary_mask(mask)

    coverage = (mask > 127).mean() * 100
    print(f"  Mask coverage: {coverage:.1f}%")

    print("\n--- LaMa Inpainting ---")
    inpainted = lama_inpaint(img, mask)
    if inpainted is None:
        print("  LaMa failed, aborting")
        return

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUTS_DIR / "lama_mask.png"), mask)
    cv2.imwrite(str(OUTPUTS_DIR / "lama_inpainted.jpg"), inpainted)

    comparison = np.hstack([
        cv2.resize(img, (400, 400)),
        cv2.resize(mask, (400, 400)),
        cv2.resize(inpainted, (400, 400)),
    ])
    cv2.imwrite(str(OUTPUTS_DIR / "lama_pipeline.jpg"), comparison)

    print(f"\n  Outputs saved to {OUTPUTS_DIR}/")
    print("    lama_pipeline.jpg  -- [original] [mask] [inpainted]")
    print("    lama_inpainted.jpg -- final result")
    print("=" * 60)


if __name__ == "__main__":
    default_image = REPO_ROOT / "sample_images" / "images" / "input3.jpg"
    default_mask = REPO_ROOT / "sample_images" / "images" / "input3_mask.jpg"
    image_arg = sys.argv[1] if len(sys.argv) > 1 else str(default_image)
    mask_arg = sys.argv[2] if len(sys.argv) > 2 else str(default_mask)
    run_inpaint(image_arg, mask_arg)
