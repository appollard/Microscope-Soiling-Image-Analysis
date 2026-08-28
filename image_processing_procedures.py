# Packages
import numpy as np
import os
import cv2
from cv2_rolling_ball import subtract_background_rolling_ball
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.ndimage import (
    gaussian_filter,
    binary_fill_holes,
    distance_transform_edt,
    label as scipy_label,
)
from skimage.filters import threshold_otsu, threshold_triangle
from skimage.measure import label, regionprops
from skimage.morphology import extrema
from skimage.segmentation import watershed, find_boundaries

import imagej
from scyjava import jimport

###################################################################################################
# SAVING FUNCTIONS
###################################################################################################


def file_to_img(file, background_colour, img_dir="microscope_images"):
    """Convert a .bmp or .jpg file into a uint8 greyscale or RGB image.

    Args:
        file (str | Path): File name or full path to the image.
        background_colour: Either "black" or "white".
        img_dir (str): Subdirectory used if only a file name is given. Defaults to
            "microscope_images".

    Returns:
        img (np.ndarray): Greyscale or coloured image as a uint8 array with white background and
            black soiling.

    Raises:
        FileNotFoundError: If the image file can not be found.
        ValueError: If the background colour is not either "black" or "white"
    """

    file = Path(file)

    if file.is_absolute():
        img_path = file
    else:
        img_path = Path(__file__).parent / img_dir / file

    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = cv2.imread(str(img_path))
    if not np.all(
        img[:, :, 0] == img[:, :, 1]
    ):  # Check if channels are identical (greyscale)
        return img  # Bypass for coloured images

    img = img[:, :, 0]
    if background_colour == "black":
        img = 255 - img
    elif background_colour == "white":
        pass
    else:
        raise ValueError("Acceptable values are 'black' or 'white'. Case sensitive.")
    return img


def img_to_file(img, filename, output_dir=None):
    """Convert uint8 greyscale image into an image (Typically png) and saves in a folder.

    Args:
        img (np.ndarray): Greyscale image as a uint8 array
        filename (str): Desired name of file, including extension
        output_dir (str | Path): Directory to save the file. Defaults to 'outputs' file.
    """

    filename = Path(filename)

    # If filename is already an absolute path, use it directly
    if filename.is_absolute():
        output_path = filename
    else:
        if output_dir is None:
            output_dir = Path(__file__).parent / "outputs"
        output_path = Path(output_dir) / filename

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Handle boolean arrays
    if np.asarray(img).dtype == bool:
        img = (np.asarray(img) * 255).astype(np.uint8)
    elif np.asarray(img).dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(output_path, img)
    return


###################################################################################################
# PRE-PROCESSING FUNCTIONS
###################################################################################################
# WIP
###################################################################################################


def subtract_noise_rolling_ball_single(img, radius):
    """Remove noise using a rolling ball, as in ImageJ.

    Args:
        img (np.ndarray): Greyscale uint8 array microscope image.
        radius (int): Radius of the rolling ball- should be larger than the largest particle.
            Defaults to 50.

    Returns:
        denoised (np.ndarray): Greyscale uint8 array microscope image with noise removed.

    """

    pad = int(radius)
    img_padded = np.pad(img, pad, mode="edge")

    result = subtract_background_rolling_ball(
        img_padded.copy(),
        radius,
        light_background=True,  # file_to_img always returns black soiling on white background
        use_paraboloid=False,
        do_presmooth=True,
    )
    denoised_padded = result[0]

    denoised = denoised_padded[pad:-pad, pad:-pad]

    return denoised


def subtract_noise_rolling_ball_full(img, radius=50):
    if img.ndim == 2:
        return subtract_noise_rolling_ball_single(img, radius)

    channels = cv2.split(img)
    corrected = [subtract_noise_rolling_ball_single(c, radius) for c in channels]
    return cv2.merge(corrected)


###################################################################################################
# ILLUMINATION NORMALISATION
###################################################################################################

# The following implementation of Retinex is sourced from:
# https://github.com/muggledy/retinex

# BSD 2-Clause License
#
# Copyright (c) 2020, Dai Yang
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


def get_gauss_kernel(sigma, dim=2):
    """1D gaussian function: G(x)=1/(sqrt{2π}σ)exp{-(x-μ)²/2σ²}. Herein, μ:=0, after
    normalizing the 1D kernel, we can get 2D kernel version by
    matmul(1D_kernel',1D_kernel), having same sigma in both directions. Note that
    if you want to blur one image with a 2-D gaussian filter, you should separate
    it into two steps(i.e. separate the 2-D filter into two 1-D filter, one column
    filter, one row filter): 1) blur image with first column filter, 2) blur the
    result image of 1) with the second row filter. Analyse the time complexity: if
    m&n is the shape of image, p&q is the size of 2-D filter, bluring image with
    2-D filter takes O(mnpq), but two-step method takes O(pmn+qmn)"""
    ksize = int(np.floor(sigma * 6) / 2) * 2 + 1  # kernel size("3-σ"法则) refer to
    # https://github.com/upcAutoLang/MSRCR-Restoration/blob/master/src/MSRCR.cpp
    k_1D = np.arange(ksize) - ksize // 2
    k_1D = np.exp(-(k_1D**2) / (2 * sigma**2))
    k_1D = k_1D / np.sum(k_1D)
    if dim == 1:
        return k_1D
    elif dim == 2:
        return k_1D[:, None].dot(k_1D.reshape(1, -1))


def gauss_blur_original(img, sigma):
    """suitable for 1 or 3 channel image"""
    row_filter = get_gauss_kernel(sigma, 1)
    t = cv2.filter2D(
        img, -1, row_filter[..., None], borderType=cv2.BORDER_REPLICATE
    )  # Default border is reflect_101, ImageJ is replicate.
    return cv2.filter2D(
        t, -1, row_filter.reshape(1, -1), borderType=cv2.BORDER_REPLICATE
    )


def gauss_blur_recursive(img, sigma):
    """refer to “Recursive implementation of the Gaussian filter”
    (doi: 10.1016/0165-1684(95)00020-E). Paper considers it faster than
    FFT(Fast Fourier Transform) implementation of a Gaussian filter.
    Suitable for 1 or 3 channel image"""
    pass


def gauss_blur(img, sigma, method="original"):
    if method == "original":
        return gauss_blur_original(img, sigma)
    elif method == "recursive":
        return gauss_blur_recursive(img, sigma)


def MultiScaleRetinex(img, sigmas=[15, 80, 250], weights=None, flag=True):
    """equal to func retinex_MSR, just remove the outer for-loop. Practice has proven
    that when MSR used in MSRCR or Gimp, we should add stretch step, otherwise the
    result color may be dim. But it's up to you, if you select to neglect stretch,
    set flag as False, have fun"""
    if weights == None:
        weights = np.ones(len(sigmas)) / len(sigmas)
    elif not abs(sum(weights) - 1) < 0.00001:
        raise ValueError("sum of weights must be 1!")
    r = np.zeros(img.shape, dtype="double")
    img = img.astype("double")
    for i, sigma in enumerate(sigmas):
        r += (
            np.log(img + 1)
            - np.log(
                gaussian_filter(img, sigma=sigma, mode="nearest") + 1
            )  # Repo uses gauss_blur- this appears to perform better.
        ) * weights[i]
    if flag:
        mmin = np.min(r, axis=(0, 1), keepdims=True)
        mmax = np.max(r, axis=(0, 1), keepdims=True)
        r = (
            (r - mmin) / (mmax - mmin) * 255
        )  # maybe indispensable when used in MSRCR or Gimp, make pic vibrant
        r = r.astype("uint8")
    return r


def retinex_gimp(img, sigmas=[12, 80, 250], dynamic=2):
    """refer to the implementation in GIMP, it improves the stretch operation based 
       on MSRCR, introduces mean and standard deviation, and a dynamic parameter to 
       eliminate chromatic aberration, experiments show that it works well. see 
       source code in https://github.com/piksels-and-lines-orchestra/gimp/blob/master \
       /plug-ins/common/contrast-retinex.c"""
    alpha = 128
    gain = 1
    offset = 0
    img = img.astype("double") + 1  #
    csum_log = np.log(np.sum(img, axis=2))
    msr = MultiScaleRetinex(img - 1, sigmas)  # -1
    r = gain * (np.log(alpha * img) - csum_log[..., None]) * msr + offset
    mean = np.mean(r, axis=(0, 1), keepdims=True)
    var = np.sqrt(np.sum((r - mean) ** 2, axis=(0, 1), keepdims=True) / r[..., 0].size)
    mmin = mean - dynamic * var
    mmax = mean + dynamic * var
    stretch = (r - mmin) / (mmax - mmin) * 255
    stretch[stretch > 255] = 255
    stretch[stretch < 0] = 0
    return stretch.astype("uint8")


###################################################################################################
# OTSU FUNCTIONS
###################################################################################################


def apply_otsu(img):
    """Generate a mask for the image with a threshold determined by the otsu function.

    Args:
        img (np.ndarray): Greyscale uint8 array with dark soiling on a light background.

    Returns:
        mask (np.ndarray): Boolean array where True indicates the presence of soiling.
            Identifies large particles.
    """

    thresh = threshold_otsu(img)
    mask = img < thresh

    return mask


###################################################################################################
# DOG FUNCTIONS
###################################################################################################


def apply_dog_triangle(img, gamma, s1=1, s2=2):
    """Generate a mask for the image using a Difference of Gaussians with triangle thresholding.

    Args:
        img (np.ndarray): Greyscale uint8 array with dark soiling on a light background.
        s1 (float): Sigma 1 for the DoG. Defaults to 1.
        s2 (float): Sigma 2 for the DoG. Defaults to 2.

    Returns:
        mask (np.ndarray): Boolean array where 1 indicates the presence of soiling.
        Identifies small particles and outlines of large particles.

    Raises:
        ValueError: If s1 >= s2, since this inverts the DoG result.
    """

    if s1 >= s2:
        raise ValueError(f"s1 ({s1}) must be less than s2 ({s2}).")

    img_float = img.astype(float)
    dog = gaussian_filter(img_float, s1) - gaussian_filter(img_float, s2)

    # Gamma adjustment — normalise to 0-1, apply gamma, scale back
    dog_min, dog_max = dog.min(), dog.max()
    dog_norm = (dog - dog_min) / (dog_max - dog_min)
    dog_gamma = np.power(dog_norm, gamma) * (dog_max - dog_min) + dog_min

    thresh = threshold_triangle(dog_gamma)
    mask = dog_gamma < thresh

    return mask


###################################################################################################
# SEGMENTED MAXIMA SEARCH WITH FIXED PROMINENCE FUNCTIONS
###################################################################################################
# WIP
###################################################################################################


def apply_fixed_prominence_maxima(img, prominence=15):
    """Segment touching particles in a grayscale microscope image using EDT watershed.

    Args:
        img (np.ndarray): 2D uint8 grayscale microscope image where dark pixels
            are particles and light pixels are background.
        prominence (int): Minimum height by which a distance transform peak must
            exceed its surroundings to seed a separate watershed region. Roughly
            corresponds to minimum particle radius in pixels. Default 15.

    Returns:
        np.ndarray: 2D boolean mask, True everywhere except on inner boundaries
            between watershed regions. AND-ing with a particle mask cuts
            agglomerated blobs at their boundary lines.
    """

    img_inv = 255 - img

    rough_fg = img < threshold_otsu(img)

    # Distance transform: peaks at particle centres, not intensity features
    dist = distance_transform_edt(rough_fg)

    # h-maxima on distance map suppresses minor peaks within one particle
    h_max = extrema.h_maxima(dist, h=prominence)
    h_max &= rough_fg

    markers, _ = scipy_label(h_max)
    labels = watershed(-dist, markers, mask=rough_fg)
    boundaries = find_boundaries(labels, mode="inner")

    return ~boundaries


###################################################################################################
# PARTICLE ANALYSIS FUNCTIONS
###################################################################################################


def fill_outlines(mask):
    """Fill any outlines present in a mask

    Args:
        mask (np.ndarray): Boolean array where True indicates the presence of soiling.

    Returns:
        mask_filled (np.ndarray): A boolean mask where any False's enclosed by True's
            are converted to True's.
    """

    mask_filled = binary_fill_holes(mask)

    return mask_filled


def identify_particles(mask_np, um_per_pixel):
    # Ensure boolean/binary
    binary = mask_np > 0

    labeled = label(binary)
    props = regionprops(labeled)

    particle_dicts = []
    for p in props:
        area_px = p.area
        area_um2 = area_px * (um_per_pixel**2)
        effective_diameter = np.sqrt(4 * area_um2 / np.pi)  # equivalent circle diameter

        particle_dicts.append(
            {
                "label": p.label,
                "area_px": area_px,
                "area_um2": area_um2,
                "effective_diameter": effective_diameter,
                "centroid": p.centroid,  # (row, col) in pixels
                "perimeter_px": p.perimeter,
                "eccentricity": p.eccentricity,
                "major_axis_length": p.major_axis_length,
                "minor_axis_length": p.minor_axis_length,
                "bbox": p.bbox,
                "coords": [
                    tuple(coord) for coord in p.coords
                ],  # (row, col) pixel list for plotting
            }
        )
    return particle_dicts


###################################################################################################
# VISUALISATION FUNCTIONS
###################################################################################################


def colourful_particle_map(particle_list, mask):
    """Generate a map of particles with different colours.

    Args:
        particle_list (list[dict]): One dict per particle, each containing:
            - 'coords' (list[tuple]): Pixel coordinates as (row, col) tuples.
            - 'centroid' (tuple[float, float]): (x, y) position in µm.
            - 'effective_diameter' (float): Diameter of equivalent circle in µm.
            - 'major_axis' (float): Major axis length of fitted ellipse in µm.
            - 'minor_axis' (float): Minor axis length of fitted ellipse in µm.
            - 'orientation' (float): Angle of major axis in radians.
            - 'pixel_count' (int): Number of pixels in the particle.
            - 'area' (float): Particle area in µm².
            - 'outline_coords' (None): Reserved for future ellipse fitting.
            - 'circumference' (None): Reserved for future ellipse fitting.
            - 'corrected_diameter' (None): Reserved for future ellipse fitting.
            - 'corrected_area' (None): Reserved for future ellipse fitting.
        mask (np.ndarray): Boolean mask where True indicates the presence of soiling

    Returns:
        map (np.ndarray): RGBA image of shape (H, W, 4) and dtype uint8, where each
            particle is coloured from a series of 6 different colours and unpopulated
            pixels are transparent (alpha = 0).

    """

    # Define RGB colour options. Last one is tranparency
    colours = np.array(
        [
            [255, 0, 0, 255],
            [0, 255, 0, 255],
            [0, 0, 255, 255],
            [255, 255, 0, 255],
            [255, 0, 255, 255],
            [0, 255, 255, 255],
            [255, 128, 0, 255],
            [128, 0, 255, 255],
            [0, 255, 128, 255],
            [255, 0, 128, 255],
            [0, 128, 255, 255],
            [128, 255, 0, 255],
        ],
        dtype=np.uint8,
    )

    # Define the map
    # + (3,) gives the tensor a depth of 4: 3 for RGB and 1 for transparency
    # (default is 0 for transparent canvas, changes to 1 when colour is assigned)
    map = np.zeros((mask.shape + (4,)), dtype=np.uint8)

    for i in range(len(particle_list)):  # For each particle
        col = colours[i % len(colours)]  # Itterate through the colours
        for j in range(len(particle_list[i]["coords"])):
            map[particle_list[i]["coords"][j]] = col  # Sets the colour of the pixel
    return map


# Overlay this on the microscope image for comparison
def show_overlay(microscope_img, mask, particle_list):
    """Generate a figure which overlays particles on a chosen image.

    Args:
        microscope_img (np.ndarray): uint8 greyscale array which has particles drawn over it.
        mask (np.ndarray): Boolean array where True inidcates the presence of soiling.
        particle_list (list[dict]): One dict per particle, each containing:
            - 'coords' (list[tuple]): Pixel coordinates as (row, col) tuples.
            - 'centroid' (tuple[float, float]): (x, y) position in µm.
            - 'effective_diameter' (float): Diameter of equivalent circle in µm.
            - 'major_axis' (float): Major axis length of fitted ellipse in µm.
            - 'minor_axis' (float): Minor axis length of fitted ellipse in µm.
            - 'orientation' (float): Angle of major axis in radians.
            - 'pixel_count' (int): Number of pixels in the particle.
            - 'area' (float): Particle area in µm².
            - 'outline_coords' (None): Reserved for future ellipse fitting.
            - 'circumference' (None): Reserved for future ellipse fitting.
            - 'corrected_diameter' (None): Reserved for future ellipse fitting.
            - 'corrected_area' (None): Reserved for future ellipse fitting.

    Displays:
        Matplotlib figure with the greyscale microscope image overlaid with a coloured RGBA
            particle map and an opacity slider.

    """

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)

    # Show the original image
    ax.imshow(microscope_img, cmap="gray", vmin=0, vmax=255)  # Assumes uint8

    # Generate the map using the particle list
    map = colourful_particle_map(particle_list, mask)

    # Show the map
    map_overlay = ax.imshow(map, alpha=0.4)

    # Slider to control map opacity
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, "Mask opacity", 0.0, 1.0, valinit=0.4)

    def update(val):
        map_overlay.set_alpha(slider.val)
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()
    plt.close("all")


def plot_hist(diameters):
    """Generate a histogram of the particle distribution.

    Args:
        diameters (np.ndarray): List of float diameters of particles, in um.

    """

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(24, 13.5))

    bins = 201
    ax.hist(diameters, bins=bins, log=True, alpha=0.6)
    ax.set_xlim((min(diameters), max(diameters)))
    ax.set_xlabel("Particle diameter (um)")
    ax.set_ylabel("Number of particles")
    ax.set_title("Frequency Plot of Particle Diameter")
    ax.legend()
    plt.show()

    return


###################################################################################################
# CLASS DEFINITION
###################################################################################################


class SoilingAnalysis:
    """Segments and analyses soiling particles from microscope images.

    Args:
        img_name (str | Path): File name or full path to the microscope image.
        background (str): Background colour, either 'black' or 'white'.
        um_per_pixel (float): Microns per pixel for size calibration.
        gamma (float): Scaling factor applied to DoG result (pixel^gamma). Defaults to 0.9
        visualiser_flag (bool): If True, displays overlay of results.
        image_dir (str | Path): Directory to search if only a filename is given. Defaults to
            'microscope_images'.
        output_dir (str | Path): Directory to output to. Defaults to 'outputs'.
    """

    def __init__(
        self,
        img_name,
        background,
        ij,
        um_per_pixel=1 / 3.156,
        visualiser_flag=True,
        gamma=0.9,
        image_dir="microscope_images",
        output_dir="outputs",
    ):
        IJ = jimport("ij.IJ")
        WindowManager = jimport("ij.WindowManager")
        RoiManager = jimport("ij.plugin.frame.RoiManager")
        self.img_name = img_name
        self.background = background
        self.um_per_pixel = um_per_pixel
        self.visualiser_flag = visualiser_flag
        self.gamma = gamma
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.threshold_method = "Triangle dark"
        self.IJ = IJ
        # Read image with white soiling, black background
        self.microscope_img = self.IJ.openImage(
            os.path.join(self.image_dir, self.img_name)
        )
        self.IJ.run(
            self.microscope_img,
            "Set Scale...",
            f"distance={self.um_per_pixel} known=1 unit=mu.m",
        )
        self.IJ.run(self.microscope_img, "8-bit", "")
        self.microscope_img.setTitle("Raw")
        self.RoiManager = RoiManager
        self.gateway = ij

    def procedure_A(self):
        """Run Procedure A: apply Otsu and DoG masks, analyse particles.

        Saves:
            'Otsu Mask.png', 'DoG Mask.png', 'Procedure A Mask.png' to output directory.

        Displays:
            Matplotlib overlay figure if visualiser_flag is True.
        """

        # Handle Background
        subtracted_background = self.microscope_img.duplicate()
        self.IJ.run(
            subtracted_background,
            "Subtract Background...",
            "rolling=50 sliding disable",
        )

        # Handle otsu
        mask1 = subtracted_background.duplicate()
        mask1.setTitle("Mask1")
        if self.background == "black":
            self.IJ.setAutoThreshold(mask1, "Otsu dark")
        elif self.background == "white":
            self.IJ.setAutoThreshold(mask1, "Otsu light")
        else:
            ValueError("Background should be 'black' or 'white'")
        self.IJ.run(mask1, "Convert to Mask", "")

        # Handle DoG
        g1 = subtracted_background.duplicate()
        g1.setTitle("G1")
        g2 = subtracted_background.duplicate()
        g2.setTitle("G2")

        sigmaG1 = 1
        sigmaG2 = 2
        self.IJ.run(g1, "Gaussian Blur...", f"sigma={sigmaG1}")
        self.IJ.run(g2, "Gaussian Blur...", f"sigma={sigmaG2}")

        ic = jimport("ij.plugin.ImageCalculator")()
        mask2 = ic.run("Subtract create", g1, g2)
        self.IJ.run(mask2, "Gamma...", f"value={self.gamma}")
        mask2.setTitle("Mask2")

        self.IJ.setAutoThreshold(mask2, self.threshold_method)
        self.IJ.run(mask2, "Convert to Mask", "")

        # Combine Masks
        binary_image = ic.run("OR create", mask1, mask2)
        binary_image.setTitle("BinaryImage")

        # Save images
        self.IJ.saveAs(mask1, "png", os.path.join(self.output_dir, "Otsu Mask"))
        self.IJ.saveAs(mask2, "png", os.path.join(self.output_dir, "DoG Mask"))
        self.IJ.saveAs(
            binary_image, "png", os.path.join(self.output_dir, "Procedure A Mask")
        )

        # Analyse particle count
        mask_np = self.gateway.py.from_java(binary_image)
        filled_mask = fill_outlines(mask_np)
        particle_dicts = identify_particles(filled_mask, self.um_per_pixel)

        # Plot histogram of diameters.
        plot_hist(np.array([p["effective_diameter"] for p in particle_dicts]))

        # Visualise the result
        if self.visualiser_flag:
            microscope_img_np = self.gateway.py.from_java(self.microscope_img)
            show_overlay(microscope_img_np, mask_np, particle_dicts)

        # Cleanup
        for img in [mask1, mask2, g1, g2, binary_image, subtracted_background]:
            if img is not None:
                img.close()

    def procedure_B(self, prominence=15, rolling_radius=50):
        """Run Procedure B: apply Otsu, DoG and fixed prominence masks, analyse particles.

        Args:
            prominence (float): Minimum peak intensity for maxima detection. Defaults to 15.
            rolling_radius (int): Radius for rolling ball background subtraction. Defaults to 50.

        Saves:
            'Otsu Mask.png', 'DoG Mask.png', 'FIxed Prominence.png', 'Procedure B Mask.png'
                to output directory.

        Displays:
            Matplotlib overlay figure if visualiser_flag is True.
        """

        denoised_img = subtract_noise_rolling_ball_full(
            self.microscope_img, rolling_radius
        )
        # denoised_img = self.microscope_img # Optional, for bug-fixing

        # Apply masks
        otsu_mask = apply_otsu(denoised_img)
        dog_mask = apply_dog_triangle(denoised_img, self.gamma)
        fixed_prominence_mask = apply_fixed_prominence_maxima(denoised_img, prominence)

        procedure_B_mask = (otsu_mask | dog_mask) & fixed_prominence_mask

        # Save masks/images
        img_to_file(self.microscope_img, "Original Image.png", self.output_dir)
        img_to_file(denoised_img, "Corrected Image.png", self.output_dir)
        img_to_file(255 - 255 * otsu_mask, "Otsu Mask.png", self.output_dir)
        img_to_file(255 - 255 * dog_mask, "DoG Mask.png", self.output_dir)
        img_to_file(
            255 - 255 * fixed_prominence_mask, "Fixed Prominence.png", self.output_dir
        )
        img_to_file(
            255 - 255 * procedure_B_mask, "Procedure B Mask.png", self.output_dir
        )

        # Analyse particle count
        filled_mask = fill_outlines(procedure_B_mask)
        particle_dicts = identify_particles(filled_mask, self.um_per_pixel)

        # Plot histogram of diameters.
        plot_hist(np.array([p["effective_diameter"] for p in particle_dicts]))

        # Visualise the result
        if self.visualiser_flag:
            show_overlay(self.microscope_img, procedure_B_mask, particle_dicts)

    def procedure_C(self, sigmas=[2, 82, 162], rolling_radius=50):
        # sigma values from "scale=240 scale_division=3" in ImageJ. Based on the formulas
        # in the source code, this yields [2, 2+240/3, 2+2*240/3], or [2, 82, 162].

        ###################################################################################################
        # WIP
        ###################################################################################################
        # Right now it just accentuates all minor flaws and makes the program return one giant particle.
        ###################################################################################################
        # denoised_img = subtract_noise_rolling_ball_full(
        #    self.microscope_img, rolling_radius
        # )
        ###################################################################################################
        denoised_img = self.microscope_img  # Optional, for bug-fixing

        # Adjust luminance
        corrected_img_rgb = retinex_gimp(denoised_img, sigmas)
        corrected_img = cv2.cvtColor(corrected_img_rgb, cv2.COLOR_BGR2GRAY)

        # Apply masks
        otsu_mask = apply_otsu(corrected_img)
        otsu_mask = fill_outlines(otsu_mask)  # Done in Cody's
        dog_mask = apply_dog_triangle(corrected_img, self.gamma)
        procedure_C_mask = otsu_mask | dog_mask

        # Save masks/images
        img_to_file(self.microscope_img, "Original Image.png", self.output_dir)
        img_to_file(corrected_img, "Corrected Image.png", self.output_dir)
        img_to_file(255 - 255 * otsu_mask, "Otsu Mask.png", self.output_dir)
        img_to_file(255 - 255 * dog_mask, "DoG Mask.png", self.output_dir)
        img_to_file(
            255 - 255 * procedure_C_mask, "Procedure C Mask.png", self.output_dir
        )

        # Analyse particle count
        particle_dicts = identify_particles(procedure_C_mask, self.um_per_pixel)

        # Plot histogram of diameters.
        plot_hist(np.array([p["effective_diameter"] for p in particle_dicts]))

        # Visualise the result
        if self.visualiser_flag:
            show_overlay(self.microscope_img, procedure_C_mask, particle_dicts)
