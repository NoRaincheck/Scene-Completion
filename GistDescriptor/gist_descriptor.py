"""A (failed) attempt at a GIST descriptor based on Gabor filter banks.

Based on the hints provided on
https://www.quora.com/Computer-Vision/What-is-a-GIST-descriptor and
http://ilab.usc.edu/siagian/Research/Gist/Gist.html

Given an input image, a GIST descriptor is computed by
1. Convolve the image with 32 Gabor filters at 4 scales and
   8 orientations, producing 32 feature maps of the same size as the
   input image.
2. Divide each feature map into 16 regions (by a 4x4 grid), and then
   average the feature values within each region.
3. Concatenate the 16 averaged values of all 32 feature maps, resulting
   in a 16x32 = 512 GIST descriptor.

Intuitively, GIST summarizes the gradient information (scales and
orientations) for different parts of an image, which provides a rough
description (the gist) of the scene.

Reference:
    Modeling the shape of the scene: a holistic representation of the
    spatial envelope.
"""

from __future__ import annotations

import glob
import os
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.filters import gabor_kernel


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def implot(im: np.ndarray, gray: bool = False) -> None:
    """Plot an OpenCV (BGR) image with matplotlib."""
    if gray or im.ndim == 2:
        plt.imshow(im.astype(np.uint8), cmap="gray")
    else:
        plt.imshow(cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.axis("off")


def ssd(imageA: np.ndarray, imageB: np.ndarray) -> float:
    """Sum of squared differences between two arrays."""
    return float(np.sum(np.square(imageA.astype("float") - imageB.astype("float"))))


def power(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Gabor power (magnitude) response of an image for one kernel."""
    # Normalize images for better comparison.
    image = (image - image.mean()) / image.std()
    return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap') ** 2 +
                   ndi.convolve(image, np.imag(kernel), mode='wrap') ** 2)


def make_kernels() -> List[Tuple[np.ndarray, str]]:
    """Build the 8 orientations x 4 frequencies Gabor filter bank."""
    results = []
    for theta in range(8):
        theta = theta / 8. * np.pi
        for frequency in (0.1, 0.2, 0.3, 0.4):
            kernel = gabor_kernel(frequency, theta=theta)
            params = 'theta=%d,\nfrequency=%.2f' % (theta * 180 / np.pi, frequency)
            results.append((kernel, params))
    return results


KERNELS = make_kernels()


def make_square(img: np.ndarray) -> np.ndarray:
    """Center-crop a grayscale image to a square with sides divisible by 4."""
    r, c = img.shape

    side4 = (min(r, c) // 4) * 4
    one_edge = side4 // 2
    img1 = img[(r // 2 - one_edge):(r // 2 + one_edge),
               (c // 2 - one_edge):(c // 2 + one_edge)]

    r, c = img1.shape
    return img1[:min(r, c), :min(r, c)]


def compute_avg(img: np.ndarray) -> np.ndarray:
    """Average each feature map over a 4x4 grid -> (4, 4) array."""
    img = make_square(img)

    r, c = img.shape

    chunks_row = np.array_split(np.arange(r), 4)
    chunks_col = np.array_split(np.arange(c), 4)

    grid_images = []
    for row in chunks_row:
        for col in chunks_col:
            grid_images.append(np.mean(img[row[0]:row[-1] + 1,
                                           col[0]:col[-1] + 1]))
    return np.array(grid_images).reshape((4, 4))


def power_single(kernel: np.ndarray, image: np.ndarray) -> np.ndarray:
    """Power response scaled back to image-like values."""
    return power(image, kernel) * 255


def get_gist_descriptor(image: np.ndarray,
                        kernels: Sequence[Tuple[np.ndarray, str]] = KERNELS,
                        max_side: int = 256) -> np.ndarray:
    """Compute the 512 dimensional GIST descriptor of a grayscale image.

    ``max_side`` caps the working resolution before the Gabor
    convolutions are applied, which keeps the descriptor cheap to
    compute for large photographs.
    """
    if max(image.shape) > max_side:
        scale = max_side / max(image.shape)
        image = cv2.resize(image, (max(4, int(round(image.shape[1] * scale))),
                                   max(4, int(round(image.shape[0] * scale)))))
    image = make_square(image.astype(np.float64) / 255.0)
    return np.array([compute_avg(power_single(kernel, image))
                     for kernel, _ in kernels]).reshape(512)


def compute_gist_descriptor(img_loc: str) -> np.ndarray:
    """Compute the GIST descriptor of an image file (grayscale)."""
    image = cv2.imread(img_loc, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise IOError(f"could not read {img_loc}")
    return get_gist_descriptor(image.astype(np.float64))


# ---------------------------------------------------------------------------
# texture classification example (adapted from the scikit-image gallery)
# ---------------------------------------------------------------------------

def compute_feats(image: np.ndarray,
                  kernels: Sequence[np.ndarray]) -> np.ndarray:
    """Mean/variance features per kernel (texture classification)."""
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    return feats


def match(feats: np.ndarray, ref_feats: np.ndarray) -> int:
    """Return index of the reference feature vector closest to ``feats``."""
    min_error = np.inf
    min_i = 0
    for i in range(ref_feats.shape[0]):
        error = np.sum((feats - ref_feats[i, :]) ** 2)
        if error < min_error:
            min_error = error
            min_i = i
    return min_i


def plot_single(kernel: np.ndarray) -> None:
    """Show one Gabor kernel."""
    plt.figure()
    plt.imshow((kernel * 255).astype(np.uint8), cmap="gray")
    plt.axis("off")


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------

def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    base_img = os.path.join(here, "..", "sample_images", "images", "input3.jpg")

    base_gist = compute_gist_descriptor(base_img)
    print("descriptor shape:", base_gist.shape)

    plt.figure()
    plt.imshow(base_gist.reshape((16, 32)), cmap="gray")
    plt.title("GIST descriptor")
    plt.axis("off")

    candidate_dir = os.path.join(here, "..", "sample_images", "images", "input3")
    candidates = sorted(glob.glob(os.path.join(candidate_dir, "*.jpg")))[:5]
    distances = []
    for cand in candidates:
        gd = compute_gist_descriptor(cand)
        distances.append(ssd(gd, base_gist))
        print(f"  {os.path.basename(cand)}: gist ssd={distances[-1]:.2f}")
    print(f"closest candidate by GIST: "
          f"{os.path.basename(candidates[int(np.argmin(distances))])}")
    plt.show()


if __name__ == "__main__":
    main()
