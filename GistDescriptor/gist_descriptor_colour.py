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

The colour variant computes the descriptor for each of the three BGR
channels independently and concatenates them (16x32x3 = 1536 values),
so all the colours are looked at rather than the grayscale image only.
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


def np2to3(im: np.ndarray) -> np.ndarray:
    """Convert a 2D (grayscale) image to a 3 channel image."""
    return np.dstack([im, im, im])


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
    """Resize an image to a square with sides divisible by 4."""
    r, c = img.shape
    side4 = (max(r, c) // 4) * 4
    return cv2.resize(img, (side4, side4))


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


def single_channel_descriptor(image: np.ndarray, max_side: int = 256) -> np.ndarray:
    """512 value GIST descriptor of one (grayscale) channel.

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
                     for kernel, _ in KERNELS]).reshape(512)


def compute_gist_descriptor3(img_loc: str, max_side: int = 256) -> np.ndarray:
    """Compute the 1536 value colour GIST descriptor of an image file."""
    image = cv2.imread(img_loc)
    if image is None:
        raise IOError(f"could not read {img_loc}")
    im0 = single_channel_descriptor(image[:, :, 0], max_side)
    im1 = single_channel_descriptor(image[:, :, 1], max_side)
    im2 = single_channel_descriptor(image[:, :, 2], max_side)
    return np.hstack([im0, im1, im2])


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------

def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    base_img = os.path.join(here, "..", "sample_images", "images", "input3.jpg")

    base_gist = compute_gist_descriptor3(base_img)
    print("colour descriptor shape:", base_gist.shape)

    plt.figure()
    plt.imshow(base_gist.reshape((16, 96)), cmap="gray")
    plt.title("colour GIST descriptor (B, G, R channels)")
    plt.axis("off")

    candidate_dir = os.path.join(here, "..", "sample_images", "images", "input3")
    candidates = sorted(glob.glob(os.path.join(candidate_dir, "*.jpg")))[:5]
    distances = []
    for cand in candidates:
        gd = compute_gist_descriptor3(cand)
        distances.append(ssd(gd, base_gist))
        print(f"  {os.path.basename(cand)}: gist ssd={distances[-1]:.2f}")
    print(f"closest candidate by colour GIST: "
          f"{os.path.basename(candidates[int(np.argmin(distances))])}")
    plt.show()


if __name__ == "__main__":
    main()
