"""Local Context Matching for scene completion.

A naive Python implementation of Local Context Matching as shown in
"Scene Completion Using Millions of Photographs"
(http://graphics.cs.cmu.edu/projects/scene-completion/).

Given an input photograph and a mask marking a region to remove, the
algorithm:
  1. crops a "local context" window around the masked hole,
  2. finds the best matching window inside a candidate photograph using
     a masked SSD search,
  3. cuts around the hole along minimal difference seams (graph cut),
  4. blends the match into the original with OpenCV seamless cloning,
  5. cleans up the paste seams with LaMa inpainting (see lama_inpaint).

Example:
    uv run python local_context_matching.py \
        --image sample_images/images/input3.jpg \
        --mask sample_images/images/input3_mask.jpg \
        --candidates-dir sample_images/images/input3 \
        --save-dir output
"""

from __future__ import annotations

import argparse
import glob
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.graph

import lama_inpaint


# ---------------------------------------------------------------------------
# ipython / plotting helpers
# ---------------------------------------------------------------------------

def implot(im: np.ndarray, gray: bool = False) -> None:
    """Plot an OpenCV (BGR) image with matplotlib."""
    if gray or im.ndim == 2:
        plt.imshow(im.astype(np.uint8), cmap="gray")
    else:
        cv_rgb = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2RGB)
        plt.imshow(cv_rgb)
    plt.axis("off")


def np2to3(im: np.ndarray) -> np.ndarray:
    """Convert a 2D (grayscale) image to a 3 channel image."""
    return np.dstack([im, im, im])


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def read_images(orig_name: str, mask_name: str, match_name: str):
    """Read the three file names and return the images for processing.

    arguments:

    orig_name : path of the original image
    mask_name : path of the mask (white = keep, black = hole to fill)
    match_name : path of the candidate image to source pixels from

    returns:
    orig, mask, match
    """
    for name in (orig_name, mask_name, match_name):
        if not os.path.isfile(name):
            raise FileNotFoundError(f"could not find image: {name}")

    orig = cv2.imread(orig_name)
    mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
    # force the mask to be black and white
    _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    match = cv2.imread(match_name)

    if orig is None or mask is None or match is None:
        raise IOError(f"failed to decode one of {orig_name}, {mask_name}, {match_name}")
    return orig, mask, match


# ---------------------------------------------------------------------------
# local context extraction
# ---------------------------------------------------------------------------

def _context_bbox(mask: np.ndarray,
                  local_context_size: int) -> Tuple[int, int, int, int]:
    """Bounding box (min_x, max_x, min_y, max_y) of the hole plus margin."""
    mask_info = np.where(mask == 0)
    if len(mask_info[0]) == 0:
        raise ValueError("the mask contains no black (masked-out) pixels")

    min_x = max(int(mask_info[0].min()) - local_context_size, 0)
    max_x = int(mask_info[0].max()) + local_context_size
    min_y = max(int(mask_info[1].min()) - local_context_size, 0)
    max_y = int(mask_info[1].max()) + local_context_size
    return min_x, max_x, min_y, max_y


def get_masked_scene(orig: np.ndarray, mask: np.ndarray,
                     local_context_size: int = 80, dilation: bool = False):
    """Crop the scene around the mask ("local context" window).

    arguments:

    orig : original image
    mask : the mask which is on the original image (0 = hole)
    local_context_size : margin in pixels added around the hole
    dilation : additionally zero out a dilated band around the hole

    returns:

    orig_scene : the cropped scene with the hole blacked out
    mask_scene : the cropped mask
    orig_scene_no_mask : the cropped scene with only the hole blacked out
                         (context intact)
    dialation_mask : 255 where context is trusted (used downstream)
    """
    orig_scene = orig.copy()
    mask_scene = mask.copy()
    orig_scene_no_mask = orig.copy()

    min_x, max_x, min_y, max_y = _context_bbox(mask, local_context_size)

    orig_scene = orig_scene[min_x:max_x, min_y:max_y]
    orig_scene_no_mask = orig_scene_no_mask[min_x:max_x, min_y:max_y]
    mask_scene = mask_scene[min_x:max_x, min_y:max_y]

    dialation_mask = (np.zeros(mask_scene.shape) + 255).astype(np.uint8)

    if dilation:
        kernel = np.ones((local_context_size, local_context_size), np.uint8)
        dialation_mask = cv2.dilate(255 - mask_scene, kernel)

    # black out the hole in both scene variants (vectorised)
    hole = mask_scene == 0
    orig_scene[hole] = 0
    orig_scene_no_mask[hole] = 0
    if dilation:
        orig_scene[dialation_mask == 0] = 0

    return orig_scene, mask_scene, orig_scene_no_mask, dialation_mask


def ssd(imageA: np.ndarray, imageB: np.ndarray, mask=None) -> Optional[float]:
    """Sum of squared differences between two images.

    Only pixels where ``imageB > 0`` are compared when no explicit mask
    is given (this matches the behaviour of the original implementation).
    """
    try:
        if mask is None:
            valid = imageB > 0
        else:
            valid = mask > 0
        diff = imageA[valid].astype("float") - imageB[valid].astype("float")
        return float(np.sum(np.square(diff)))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# best match search
# ---------------------------------------------------------------------------

def _context_mask(scene: np.ndarray) -> np.ndarray:
    """Binary float mask of the valid (non hole) context pixels."""
    return (cv2.cvtColor(scene.astype(np.uint8), cv2.COLOR_BGR2GRAY) > 0).astype(np.float32)


def find_scene_bruteforce(orig_scene: np.ndarray, match: np.ndarray):
    """Reference brute force masked-SSD search (slow, kept for tests).

    Returns (best_x, best_y, best_sample) where best_x/best_y are the
    row/column offsets of the best matching window inside ``match``.
    """
    ir, ic, _ = orig_scene.shape
    r, c, _ = match.shape
    if ir > r or ic > c:
        raise ValueError("match image is smaller than the local context window")

    min_ssd = None
    best_sample = None
    best_x = best_y = None
    for x in range(r - ir + 1):          # row offset
        for y in range(c - ic + 1):      # column offset
            imageA = match[x:x + ir, y:y + ic]
            current_ssd = ssd(imageA, orig_scene)
            if current_ssd is None:
                continue
            if min_ssd is None or min_ssd > current_ssd:
                min_ssd = current_ssd
                best_sample = imageA.copy()
                best_x, best_y = x, y
    if best_sample is None:
        raise ValueError("no overlapping context between scene and match")
    return best_x, best_y, best_sample


def find_scene(orig_scene: np.ndarray, match: np.ndarray):
    """Find the best match for the masked scene within the match image.

    Equivalent to a brute force masked SSD search over every window of
    the match image, but computed with ``cv2.matchTemplate`` so that the
    full search space is covered instantly instead of scanning only the
    first row of positions.

    returns:

    best_x : row coordinate of best matching window
    best_y : column coordinate of best matching window
    match_scene : the best matching window
    """
    ir, ic, _ = orig_scene.shape
    r, c, _ = match.shape
    if ir > r or ic > c:
        raise ValueError(
            f"match image {match.shape[:2]} is smaller than the "
            f"local context window {orig_scene.shape[:2]}")

    mask = _context_mask(orig_scene)
    if not mask.any():
        raise ValueError("local context contains no valid (non black) pixels")

    result = cv2.matchTemplate(match.astype(np.float32),
                               orig_scene.astype(np.float32),
                               cv2.TM_SQDIFF, mask=mask)
    _, _, min_loc, _ = cv2.minMaxLoc(result)
    # min_loc is (x=column, y=row); keep the original row/col convention
    best_y, best_x = min_loc
    best_sample = match[best_x:best_x + ir, best_y:best_y + ic].copy()
    return best_x, best_y, best_sample


def score_candidate(orig_scene: np.ndarray, match: np.ndarray) -> float:
    """Masked context SSD of the best alignment (lower is better).

    Used to rank multiple candidate photographs against each other.
    """
    ir, ic, _ = orig_scene.shape
    if ir > match.shape[0] or ic > match.shape[1]:
        return float("inf")
    mask = _context_mask(orig_scene)
    if not mask.any():
        return float("inf")
    result = cv2.matchTemplate(match.astype(np.float32),
                               orig_scene.astype(np.float32),
                               cv2.TM_SQDIFF, mask=mask)
    return float(result.min())


# ---------------------------------------------------------------------------
# seam cutting
# ---------------------------------------------------------------------------

def _edge_points(lengths: Sequence[int], fixed: int, horizontal: bool,
                 shape: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Build (row, col) points along an image edge segment.

    ``horizontal=True`` runs along the top/bottom edge at row ``fixed``;
    otherwise along the left/right edge at column ``fixed``.
    """
    points = []
    for v in lengths:
        v = int(v)
        if v < 0:
            continue
        row, col = (fixed, v) if horizontal else (v, fixed)
        if 0 <= row < shape[0] and 0 <= col < shape[1]:
            points.append((row, col))
    return points


def create_seam_cut(orig_scene: np.ndarray, mask_scene: np.ndarray,
                    match_scene: Optional[np.ndarray] = None,
                    orig_scene_no_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Cut around the hole using minimal difference seams (graph cut).

    Four seams are traced with Dijkstra (skimage MCP) through the
    per-pixel absolute difference map, joining the edge segments that
    surround the hole. The enclosed area (hole + seams) is then
    recovered with a flood fill.

    returns:
    mask_seam : 3 channel array, 255 where replacement happens
    """
    if match_scene is None:
        match_scene = np.ones_like(orig_scene) * 255
    if orig_scene_no_mask is None:
        orig_scene_no_mask = np.ones_like(orig_scene) * 255

    # cast to float BEFORE differencing: uint8 subtraction wraps around
    diff = np.absolute(match_scene.astype(np.float64) - orig_scene.astype(np.float64))
    diff_gray = cv2.cvtColor(diff.astype(np.float32), cv2.COLOR_BGR2GRAY).astype(np.float64)

    # seams may never pass through the hole itself
    diff_gray[mask_scene == 0] = np.inf

    mask_info = np.where(mask_scene == 0)
    h, w = diff_gray.shape
    min_x, max_x = int(mask_info[0].min()), int(mask_info[0].max())
    min_y, max_y = int(mask_info[1].min()), int(mask_info[1].max())

    adj = 10

    NW_top = _edge_points(range(0, min_y - adj), 0, True, diff_gray.shape)
    NW_left = _edge_points(range(0, min_x - adj), 0, False, diff_gray.shape)

    NE_top = _edge_points(range(max_y + adj, w), 0, True, diff_gray.shape)
    NE_right = _edge_points(range(0, min_x - adj), w - 1, False, diff_gray.shape)

    SW_left = _edge_points(range(0, min_y - adj), h - 1, True, diff_gray.shape)
    SW_bot = _edge_points(range(max_x + adj, h), 0, False, diff_gray.shape)

    SE_right = _edge_points(range(max_y + adj, w), h - 1, True, diff_gray.shape)
    SE_bot = _edge_points(range(max_x + adj, h), w - 1, False, diff_gray.shape)
    bottom_right = SE_right + SE_bot

    def trace(starts, ends, out):
        """Trace up to 10 random minimum cost seams and paint them."""
        starts, ends = list(starts), list(ends)
        if not starts or not ends:
            return False
        costMCP = skimage.graph.MCP(diff_gray, fully_connected=True)
        costMCP.find_costs(starts=starts, ends=ends)
        painted = False
        for _ in range(10):
            end = random.choice(ends)
            try:
                path = costMCP.traceback(end)
            except Exception:
                continue
            if not path:
                continue
            for x, y in path:
                out[x, y] = 255
            painted = True
        return painted

    diff_path = np.zeros(diff_gray.shape, np.uint8)
    # four cuts around the hole: across the top / down the left /
    # down the right / across the bottom
    trace(NW_left, NE_right, diff_path)
    trace(NW_top, SW_bot, diff_path)
    trace(NE_top, SE_bot, diff_path)
    trace(SW_left, bottom_right, diff_path)

    # flood fill from a point inside the hole to close the seam loop.
    hole_rows, hole_cols = np.where((orig_scene_no_mask[:, :, 0] == 0) &
                                    (diff_path == 0))
    if len(hole_rows) == 0:
        hole_rows, hole_cols = np.where(orig_scene_no_mask[:, :, 0] == 0)
    if len(hole_rows) == 0:
        raise ValueError("cannot find a seed point inside the hole for flood fill")

    diff_fill = diff_path.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    rnd = random.randrange(len(hole_rows))
    seed = (int(hole_cols[rnd]), int(hole_rows[rnd]))
    cv2.floodFill(diff_fill, ff_mask, seed, 255, loDiff=0, upDiff=0)

    # erode and dilate the result to clean up speckles
    kernel = np.ones((5, 5), np.uint8)
    diff_fill = cv2.erode(diff_fill, kernel, iterations=2)
    diff_fill = cv2.dilate(diff_fill, kernel, iterations=2)

    # blur and threshold to slightly feather the seam
    diff_fill = cv2.blur(diff_fill, (10, 10))
    _, diff_fill = cv2.threshold(diff_fill, 5, 255, cv2.THRESH_BINARY)
    return np2to3(diff_fill)


# ---------------------------------------------------------------------------
# compositing
# ---------------------------------------------------------------------------

def composite_scene(orig_scene: np.ndarray, mask_seam: np.ndarray,
                    match_scene: np.ndarray, dialation_mask: np.ndarray,
                    orig_scene1: np.ndarray, method: str = "paste",
                    repeat: int = 1) -> np.ndarray:
    """Combine the original and matched scenes based on the seam mask.

    method='paste'          straight copy inside the seam region
    method='alphablend'     feathered blend along the seam
    method='seamlessclone'  Poisson blending (cv2.seamlessClone) with an
                            alpha blended refinement pass
    """
    avg_pixel = np.mean(orig_scene1[orig_scene1 != 0])

    output = np.zeros(orig_scene.shape, np.float64)

    if method == "seamlessclone":
        hh, ww = match_scene.shape[:2]
        center = (ww // 2, hh // 2)  # must be integers for seamlessClone

        # create plain white mask
        mask = np.full(match_scene.shape, 255, match_scene.dtype)

        orig_scene_impute = orig_scene.copy()
        orig_scene_impute[mask_seam == 255] = avg_pixel

        output_blend = cv2.seamlessClone(match_scene.astype(np.uint8),
                                         orig_scene_impute.astype(np.uint8),
                                         mask, center, cv2.NORMAL_CLONE).astype(np.float64)

        # blur the seam mask and use it as alpha for feathering
        dilation_mask = mask_seam.astype(np.float64)
        dilation_mask = cv2.GaussianBlur(dilation_mask, (101, 101), 0)
        dilation_mask = dilation_mask / 255.0

        for _ in range(repeat + 10):
            # layered alpha blend between the clone and the original
            orig_scene_impute = orig_scene.astype(np.float64)
            orig_scene_impute[mask_seam == 255] = output_blend[mask_seam == 255]
            output_blend = cv2.add(cv2.multiply(output_blend, dilation_mask),
                                   cv2.multiply(orig_scene_impute, 1 - dilation_mask))

        orig_scene_impute = orig_scene.copy()
        orig_scene_impute[mask_seam == 255] = output_blend[mask_seam == 255]
        output = cv2.seamlessClone(match_scene.astype(np.uint8),
                                   np.clip(output_blend, 0, 255).astype(np.uint8),
                                   mask, center, cv2.NORMAL_CLONE)
        output = output.astype(np.float64)

    elif method == "paste":
        output[mask_seam == 0] = orig_scene[mask_seam == 0]
        output[mask_seam != 0] = match_scene[mask_seam != 0]

    elif method == "alphablend":
        # feathered blend using the seam mask as alpha
        alpha = mask_seam.astype(np.float64)
        alpha = cv2.GaussianBlur(alpha, (31, 31), 0) / 255.0
        output = (orig_scene.astype(np.float64) * (1 - alpha) +
                  match_scene.astype(np.float64) * alpha)

    else:
        output[mask_seam == 0] = orig_scene[mask_seam == 0]
        output[mask_seam != 0] = match_scene[mask_seam != 0]

    return np.clip(output, 0, 255)


def composite(orig: np.ndarray, com_scene: np.ndarray, mask: np.ndarray,
              local_context_size: int = 80) -> np.ndarray:
    """Paste the completed local context back into the full size image."""
    min_x, max_x, min_y, max_y = _context_bbox(mask, local_context_size)

    orig_new = orig.copy().astype(np.float64)
    orig_new[min_x:max_x, min_y:max_y] = com_scene
    return np.clip(orig_new, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# lama seam cleanup
# ---------------------------------------------------------------------------

def build_seam_band_mask(mask_seam: np.ndarray, band: int = 12) -> np.ndarray:
    """Ring mask straddling the boundary of the replaced region.

    ``mask_seam`` marks where match content was pasted over the original
    (255 = replaced).  The visible paste seams live exactly on that
    boundary, so we cover a band of ``band`` pixels outside it and about
    half that inside it: enough for LaMa to re-synthesise the transition
    without repainting the whole fill.
    """
    m = mask_seam if mask_seam.ndim == 2 else cv2.cvtColor(
        mask_seam.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    m = ((m > 0) * 255).astype(np.uint8)

    k_out = 2 * band + 1                      # dilate radius ~ band
    k_in = max(3, band | 1)                   # erode radius ~ band / 2
    outer = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                    (k_out, k_out)))
    inner = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                   (k_in, k_in)))
    return ((outer > 0) & (inner == 0)).astype(np.uint8) * 255


def inpaint_seams_lama(stages: Dict[str, object], local_context_size: int = 55,
                       band: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """Clean up the composited output by inpainting its seams with LaMa.

    A seam band mask is derived from the pipeline's seam cut, lifted back
    to full resolution coordinates and handed to LaMa; the result is
    feather-blended into the output so pixels away from the seams stay
    exactly as they were.

    returns:
    (cleaned output, full resolution seam band mask)
    """
    output = np.clip(stages["output"], 0, 255).astype(np.uint8)
    h, w = output.shape[:2]

    band_scene = build_seam_band_mask(stages["mask_seam"], band)

    # the hole rim is always a material boundary (pasted match content
    # against the original photograph); ring it too so cleanup still
    # happens when the seam cut degenerated into a solid mask_seam
    replaced_hole = (stages["mask_scene"] == 0).astype(np.uint8) * 255
    band_scene = cv2.bitwise_or(band_scene,
                                build_seam_band_mask(replaced_hole, band))
    min_x, max_x, min_y, max_y = _context_bbox(stages["mask"],
                                               local_context_size)

    # lift the scene-space band into a full resolution mask (the crop may
    # run past the image edges, hence the clipped slices)
    band_full = np.zeros((h, w), np.uint8)
    rows = min(max_x, h) - min_x
    cols = min(max_y, w) - min_y
    band_full[min_x:min_x + rows, min_y:min_y + cols] = \
        band_scene[:rows, :cols]

    if not (band_full > 0).any():
        print("  LaMa: empty seam band, skipping inpainting")
        return output, band_full

    print(f"  LaMa: inpainting seam band "
          f"({int((band_full > 0).sum())} px)")
    cleaned = lama_inpaint.lama_inpaint(output, band_full)
    if cleaned is None:
        return output, band_full

    # feathered blend: untouched pixels stay bit-exact, only the band is
    # replaced by the inpainted content
    kernel = 2 * band + 1
    alpha = cv2.GaussianBlur(band_full, (kernel, kernel), 0)[..., None] / 255.0
    blended = output.astype(np.float64) * (1 - alpha) + \
        cleaned.astype(np.float64) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8), band_full


# ---------------------------------------------------------------------------
# high level pipeline
# ---------------------------------------------------------------------------

def scene_completion_pipeline(orig_name: str, mask_name: str, match_name: str,
                              local_context_size: int = 55,
                              dilation: bool = True,
                              lama: bool = True,
                              lama_band: int = 12) -> Dict[str, object]:
    """Run the full pipeline and return every intermediate stage.

    When ``lama`` is true the composited output gets a final pass in which
    the paste seams are re-synthesised with LaMa inpainting (``lama_band``
    controls the width of the re-inpainted band around each seam).
    """
    orig, mask, match = read_images(orig_name, mask_name, match_name)
    orig_scene, mask_scene, orig_scene_no_mask, dialation_mask = get_masked_scene(
        orig, mask, local_context_size, dilation=dilation)
    best_x, best_y, match_scene = find_scene(orig_scene, match)
    mask_seam = create_seam_cut(orig_scene, mask_scene, match_scene, orig_scene_no_mask)

    com_scene = composite_scene(orig_scene_no_mask, mask_seam, match_scene,
                                dialation_mask, orig_scene, method="seamlessclone")
    paste_com = composite_scene(orig_scene_no_mask, mask_seam, match_scene,
                                dialation_mask, orig_scene, method="paste")
    paste_com1 = composite_scene(orig_scene_no_mask, 255 - np2to3(mask_scene), match_scene,
                                 dialation_mask, orig_scene, method="paste")
    paste_com2 = composite_scene(orig_scene_no_mask, 255 - np2to3(mask_scene), match_scene,
                                 dialation_mask, orig_scene, method="seamlessclone")
    paste_com3 = composite_scene(orig_scene_no_mask, mask_seam, paste_com2,
                                 dialation_mask, orig_scene, method="paste")
    paste_com4 = composite_scene(orig_scene_no_mask, 255 - np2to3(mask_scene),
                                 paste_com3.astype(np.uint8),
                                 dialation_mask, orig_scene, method="seamlessclone")
    output = composite(orig, paste_com3, mask, local_context_size)

    stages = {
        "orig": orig, "mask": mask, "match": match,
        "orig_scene": orig_scene, "mask_scene": mask_scene,
        "orig_scene_no_mask": orig_scene_no_mask,
        "dialation_mask": dialation_mask,
        "best_x": best_x, "best_y": best_y,
        "match_scene": match_scene, "mask_seam": mask_seam,
        "com_scene": com_scene, "paste_com": paste_com,
        "paste_com1": paste_com1, "paste_com2": paste_com2,
        "paste_com3": paste_com3, "paste_com4": paste_com4,
        "output": output,
    }

    if lama:
        output_lama, seam_band_mask = inpaint_seams_lama(
            stages, local_context_size, lama_band)
        stages["seam_band_mask"] = seam_band_mask
        stages["output_lama"] = output_lama

    return stages


def local_context_match(orig_name: str, mask_name: str, match_name: str,
                        local_context_size: int = 55,
                        method: str = 'seamlessclone',
                        dilation: bool = True,
                        lama: bool = True,
                        lama_band: int = 12) -> np.ndarray:
    """Complete the ``mask_name`` region of ``orig_name`` using ``match_name``.

    Sample usage::

        output = local_context_match("source.jpg", "source_mask.jpg", "match.jpg", 60)
    """
    stages = scene_completion_pipeline(orig_name, mask_name, match_name,
                                       local_context_size, dilation,
                                       lama=lama, lama_band=lama_band)
    return stages.get("output_lama", stages["output"])


def pick_best_candidate(orig_name: str, mask_name: str,
                        candidates: Sequence[str],
                        local_context_size: int = 55) -> Tuple[str, float]:
    """Rank candidate images by masked context SSD and return the best."""
    orig, mask, _ = read_images(orig_name, mask_name, candidates[0])
    orig_scene, _, _, _ = get_masked_scene(orig, mask, local_context_size)
    best_name, best_score = None, float("inf")
    for name in candidates:
        cand = cv2.imread(name)
        if cand is None:
            continue
        score = score_candidate(orig_scene, cand)
        print(f"  candidate {os.path.basename(name)}: ssd={score:.4e}")
        if score < best_score:
            best_name, best_score = name, score
    if best_name is None:
        raise ValueError("no readable candidate images found")
    return best_name, best_score


def save_stages(stages: Dict[str, object], save_dir: str) -> None:
    """Write every pipeline stage to ``save_dir`` as jpg files."""
    os.makedirs(save_dir, exist_ok=True)
    for name, im in stages.items():
        if isinstance(im, np.ndarray):
            cv2.imwrite(os.path.join(save_dir, f"{name}.jpg"),
                        np.clip(im, 0, 255).astype(np.uint8))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scene completion via local context matching.")
    parser.add_argument("--image", required=True, help="original photograph")
    parser.add_argument("--mask", required=True,
                        help="mask image, black = region to fill")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--match", help="candidate image to source pixels from")
    group.add_argument("--candidates-dir",
                       help="directory of candidates; the best scoring one is used")
    parser.add_argument("--context-size", type=int, default=55,
                        help="local context margin in pixels (default 55)")
    parser.add_argument("--no-lama", action="store_true",
                        help="skip the final LaMa seam inpainting pass")
    parser.add_argument("--lama-band", type=int, default=12,
                        help="width in pixels of the LaMa seam band (default 12)")
    parser.add_argument("--save-dir", default="output",
                        help="where to write the result and diagnostics")
    parser.add_argument("--quiet", action="store_true",
                        help="do not print per-candidate scores")
    args = parser.parse_args(argv)

    random.seed(0)

    if args.candidates_dir:
        candidates = sorted(glob.glob(os.path.join(args.candidates_dir, "*.jpg")))
        if not candidates:
            parser.error(f"no .jpg candidates in {args.candidates_dir}")
        if args.quiet:
            import contextlib
            with contextlib.redirect_stdout(None):
                best, score = pick_best_candidate(args.image, args.mask, candidates,
                                                  args.context_size)
        else:
            best, score = pick_best_candidate(args.image, args.mask, candidates,
                                              args.context_size)
        print(f"selected candidate: {os.path.basename(best)} "
              f"(context SSD {score:.3e})")
        match_name = best
    else:
        match_name = args.match

    stages = scene_completion_pipeline(args.image, args.mask, match_name,
                                       args.context_size,
                                       lama=not args.no_lama,
                                       lama_band=args.lama_band)
    save_stages(stages, args.save_dir)
    print(f"wrote {os.path.join(args.save_dir, 'output.jpg')}")
    if "output_lama" in stages:
        print(f"wrote {os.path.join(args.save_dir, 'output_lama.jpg')} "
              f"(LaMa seam cleanup)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
