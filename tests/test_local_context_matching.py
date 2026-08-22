"""Tests for the local context matching pipeline.

Run with: uv run pytest
"""

import os

import cv2
import numpy as np
import pytest

import local_context_matching as lcm
import lama_inpaint

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG3 = os.path.join(REPO, "sample_images", "images", "input3.jpg")
MASK3 = os.path.join(REPO, "sample_images", "images", "input3_mask.jpg")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_synthetic_case():
    """Small synthetic scene + match with a known optimal alignment."""
    rng = np.random.default_rng(42)
    # match image: smooth gradient + noise, 60x80
    match = np.tile(np.linspace(30, 220, 80, dtype=np.float32), (60, 1))
    match = np.dstack([match] * 3)
    match += rng.normal(0, 2, match.shape)
    match = np.clip(match, 0, 255).astype(np.uint8)

    # scene is a window of the match taken at row 10, col 20
    scene = match[10:40, 20:60].copy()
    return scene, match


# ---------------------------------------------------------------------------
# unit tests
# ---------------------------------------------------------------------------

def test_find_scene_matches_bruteforce_reference():
    """find_scene must agree with the brute force masked-SSD reference."""
    scene, match = make_synthetic_case()

    bx, by, sample = lcm.find_scene(scene, match)
    fx, fy, fsample = lcm.find_scene_bruteforce(scene, match)

    assert (bx, by) == (fx, fy) == (10, 20)
    assert np.array_equal(sample, fsample)


def test_find_scene_handles_masked_hole():
    """The search must ignore hole pixels (all-zero region)."""
    rng = np.random.default_rng(7)
    match = rng.integers(0, 255, (50, 50, 3), dtype=np.uint8)

    scene = match[5:25, 5:35].copy()
    scene[:, 15:] = 0  # punch a hole in the right half of the scene

    best_x, best_y, sample = lcm.find_scene(scene, match)
    # the true window must win because the visible half matches exactly
    # while random windows differ; verify returned crop matches the scene
    # on its valid pixels better than a deliberately wrong window does.
    valid = scene > 0
    good = float(np.sum((sample.astype(float)[valid] - scene.astype(float)[valid]) ** 2))
    bad_window = match[30:, 10:40][:scene.shape[0], :scene.shape[1]]
    bad = float(np.sum((bad_window.astype(float)[valid] - scene.astype(float)[valid]) ** 2))
    assert good < bad


def test_np2to3_replicates_channels():
    gray = (np.arange(12).reshape(3, 4) * 20).astype(np.uint8)
    rgb = lcm.np2to3(gray)
    assert rgb.shape == (3, 4, 3)
    assert np.array_equal(rgb[:, :, 0], gray)
    assert np.array_equal(rgb[:, :, 2], gray)


def test_ssd_ignores_black_pixels_of_imageB():
    a = np.full((2, 2, 3), 100, np.uint8)
    b = np.zeros((2, 2, 3), np.uint8)
    b[0, 0] = 100
    # only pixel (0,0) is compared -> zero difference
    assert lcm.ssd(a, b) == 0.0


def test_get_masked_scene_crops_and_blacks_out_hole():
    # gradient image so cropped pixels can be verified exactly
    orig = np.dstack([np.arange(200 * 200, dtype=np.uint16).reshape(200, 200) % 256] * 3)
    orig = orig.astype(np.uint8)
    mask = np.full((200, 200), 255, np.uint8)
    mask[95:115, 95:115] = 0  # small interior hole at centre

    orig_scene, mask_scene, no_mask, dialation_mask = lcm.get_masked_scene(
        orig, mask, local_context_size=30)

    # crop = hole bbox grown by the context margin (exclusive upper bound
    # matches the original implementation), clamped to the image
    assert orig_scene.shape == (79, 79, 3)
    assert mask_scene.shape == (79, 79)
    hole = mask_scene == 0
    assert hole.sum() == 20 * 20
    assert not orig_scene[hole].any()      # hole blacked out
    assert not no_mask[hole].any()
    # context pixels must be an exact copy of the original crop
    assert np.array_equal(orig_scene[~hole], orig[65:144, 65:144][~hole])
    assert np.array_equal(dialation_mask, np.full((79, 79), 255, np.uint8))


def test_create_seam_cut_masks_entire_hole():
    rng = np.random.default_rng(3)
    scene = rng.integers(0, 255, (120, 140, 3), dtype=np.uint8)
    match = rng.integers(0, 255, (120, 140, 3), dtype=np.uint8)
    mask_scene = np.zeros((120, 140), np.uint8)
    mask_scene[40:80, 45:95] = 255  # interior hole

    seam = lcm.create_seam_cut(scene.copy(), mask_scene, match,
                               scene.copy())
    hole = mask_scene == 0
    seam_gray = cv2.cvtColor(seam.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    coverage = (seam_gray[hole] > 0).mean()
    assert coverage > 0.99, f"seam covers only {coverage:.1%} of the hole"


def test_build_seam_band_mask_is_a_boundary_ring():
    """Ring mode must hug the replaced-region boundary on both sides."""
    replaced = np.zeros((100, 100), np.uint8)
    replaced[30:70, 35:65] = 255

    band = lcm.build_seam_band_mask(lcm.np2to3(replaced), band=6)
    g = band

    assert g.shape == (100, 100)
    assert g.dtype == np.uint8
    assert g[50, 50] == 0        # deep inside the replaced region: untouched
    assert g[2, 2] == 0          # far outside: untouched
    assert g[29, 50] == 255      # just outside the boundary (original side)
    assert g[71, 50] == 255      # just outside on the other side
    assert g[31, 50] == 255      # just inside (match side)
    # ring must be thin relative to the region it surrounds
    assert (g > 0).sum() < 0.25 * g.size


def test_build_seam_band_mask_solid_fills_region():
    """Solid mode must cover the whole replaced region plus its margin."""
    replaced = np.zeros((100, 100), np.uint8)
    replaced[30:70, 35:65] = 255

    solid = lcm.build_seam_band_mask(replaced, band=6, solid=True)

    assert solid.shape == (100, 100)
    assert (solid[30:70, 35:65] == 255).all()   # interior fully covered
    assert solid[29, 50] == 255                 # margin grown outside
    assert solid[71, 50] == 255
    assert solid[22, 50] == 0                   # beyond ~band: untouched
    assert solid[2, 2] == 0


def test_adaptive_band_width_grows_with_hole_size():
    small = np.full((200, 200), 255, np.uint8)
    small[90:110, 90:110] = 0                       # 20 px hole
    assert lcm._adaptive_band_width(small, min_band=12,
                                    scale=0.25) == 12   # clamped to minimum

    big = np.full((400, 400), 255, np.uint8)
    big[100:300, 150:250] = 0                       # 200 px hole
    assert lcm._adaptive_band_width(big, min_band=12,
                                    scale=0.25) == 50   # 0.25 * 200

    huge = np.full((1000, 1000), 255, np.uint8)
    huge[100:900, 100:900] = 0                      # 800 px hole
    assert lcm._adaptive_band_width(huge, min_band=12,
                                    scale=0.25) == 64   # clamped to maximum


# ---------------------------------------------------------------------------
# integration tests on the bundled samples
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_full_pipeline_on_sample_image():
    stages = lcm.scene_completion_pipeline(IMG3, MASK3,
                                           os.path.join(REPO, "sample_images",
                                                        "images", "input3",
                                                        "result_img001.jpg"))
    orig, out, mask = stages["orig"], stages["output"], stages["mask"]
    hole, keep = mask == 0, mask == 255

    assert out.shape == orig.shape
    assert out.dtype == np.uint8
    # the hole must be filled with new content
    assert np.abs(out[hole].astype(float) - orig[hole].astype(float)).mean() > 1
    # the kept context must be almost untouched
    assert np.abs(out[keep].astype(float) - orig[keep].astype(float)).mean() < 20


@pytest.mark.slow
@pytest.mark.skipif(not lama_inpaint.DEFAULT_MODEL_PATH.exists(),
                    reason="models/lama.onnx not present")
def test_lama_seam_cleanup_only_touches_the_band():
    """LaMa cleanup must alter pixels near seams and leave the rest exact."""
    stages = lcm.scene_completion_pipeline(IMG3, MASK3,
                                           os.path.join(REPO, "sample_images",
                                                        "images", "input3",
                                                        "result_img001.jpg"))
    out, cleaned = stages["output"], stages["output_lama"]
    band = stages["seam_band_mask"]

    assert cleaned.shape == out.shape and cleaned.dtype == np.uint8
    assert band.shape == out.shape[:2]
    assert (band > 0).any()

    # pixels well away from any seam must be bit-identical
    far = cv2.erode((band == 0).astype(np.uint8),
                    np.ones((31, 31), np.uint8)) > 0
    assert far.any()
    assert np.array_equal(out[far], cleaned[far])

    # ...and something near the seams must actually have been re-synthesised
    assert (out != cleaned).any()
