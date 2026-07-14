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
from skimage.morphology import flood_fill
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

###################################################################################################
# SAVING FUNCTIONS
###################################################################################################


def file_to_img(file, background_colour, img_dir="microscope_images"):
    """Convert a .bmp or .jpg file into a uint8 greyscale image.

    Args:
        file (str | Path): File name or full path to the image.
        background_colour: Either "black" or "white".
        img_dir (str): Subdirectory used if only a file name is given. Defaults to
            "microscope_images".

    Returns:
        grey_img (np.ndarray): Greyscale image as a uint8 array with white background and
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
    if img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img

    if background_colour == "black":
        gray_img = 255 - gray_img
    elif background_colour == "white":
        pass
    else:
        raise ValueError("Acceptable values are 'black' or white'. Case sensitive.")
    return gray_img


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


def subtract_noise_rolling_ball(img, radius=50):
    """Remove noise using a rolling ball, as in ImageJ.

    Args:
        img (np.ndarray): Greyscale uint8 array microscope image.
        img_background (str): Either 'black' or 'white'.
        radius (int): Radius of the rolling ball- should be larger than the largest particle.
            Defaults to 50.

    Returns:
        denoised (np.ndarray): Greyscale uint8 array microscope image with noise removed.

    """

    denoised, _ = subtract_background_rolling_ball(
        img.copy(),
        radius,
        light_background=True,  # file_to_img always returns black soiling on white background
        use_paraboloid=False,
        do_presmooth=True,
    )
    return denoised


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

    from skimage.morphology import extrema
    from skimage.segmentation import find_boundaries

    img_inv = (255 - img).astype(np.float32)  # dark particles become bright peaks

    # h-maxima finds every point prominence higher than surroundings
    h_maxima = extrema.h_maxima(img_inv, h=prominence)
    markers, _ = scipy_label(h_maxima)
    labels = watershed(-img_inv, markers)
    boundaries = find_boundaries(labels, mode="outer")

    # 1 everywhere except boundaries
    return boundaries ^ True


###################################################################################################
# ILLUMINATION NORMALISATION
###################################################################################################
# WIP
###################################################################################################


from skimage.exposure import equalize_adapthist


def correct_illumination(img):
    # Find something that uses retinex online

    return img


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


def identify_particles(mask, um_per_pixel):
    """Identify the particles in a mask and key information about them.

    Args:
        mask (np.ndarray): Boolean array where True indicates the presence of soiling.
        um_per_pixel (float): Conversion ratio with units microns/pixel.

    Returns:
        particle_dicts (list[dict]): One dict per particle, each containing:
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
    """

    # Label connected regions (each particle gets a unique integer)
    labelled = label(mask)
    props = regionprops(labelled, intensity_image=mask)

    particle_dicts = []
    for p in props:
        # coords from regionprops are (row, col) tuples, same as original
        coords = [tuple(c) for c in p.coords.tolist()]

        particle_dicts.append(
            {
                "coords": coords,
                "centroid": (
                    p.centroid[1] * um_per_pixel,  # x in um
                    p.centroid[0] * um_per_pixel,
                ),  # y in um
                "effective_diameter": p.equivalent_diameter_area * um_per_pixel,
                "major_axis": p.axis_major_length * um_per_pixel,
                "minor_axis": p.axis_minor_length * um_per_pixel,
                "orientation": p.orientation,
                "pixel_count": len(coords),
                "area": p.area * (um_per_pixel**2),
                # kept as None — no ellipse method being run
                "outline_coords": None,
                "circumference": None,
                "corrected_diameter": None,
                "corrected_area": None,
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
        mask (np.ndarray) Boolean mask where True indicates the presence of soiling

    Returns:
        map (np.ndarray): RGBA image of shape (H, W, 4) and dtype uint8, where each
            particle is coloured from a series of 6 different colours and unpopulated
            pixels are transparent (alpha = 0).

    """

    # Define RGB colour options. Last one is tranparency
    colours = np.uint8(
        [
            (255, 0, 0, 255),
            (0, 255, 0, 255),
            (0, 0, 255, 255),
            (255, 255, 0, 255),
            (255, 0, 255, 255),
            (0, 255, 255, 255),
        ]
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
        um_per_pixel,
        visualiser_flag,
        gamma=0.9,
        image_dir="microscope_images",
        output_dir="outputs",
    ):
        self.img_name = img_name
        self.background = background
        self.um_per_pixel = um_per_pixel
        self.visualiser_flag = visualiser_flag
        self.gamma = gamma
        self.image_dir = image_dir
        self.output_dir = output_dir
        # Read image with white soiling, black background
        self.microscope_img = file_to_img(img_name, background, self.image_dir)

    def procedure_A(self, rolling_radius=50):
        """Run Procedure A: apply Otsu and DoG masks, analyse particles.

        Saves:
            'Otsu Mask.png', 'DoG Mask.png', 'Procedure A Mask.png' to output directory.

        Displays:
            Matplotlib overlay figure if visualiser_flag is True.
        """

        ###################################################################################################
        # WIP
        ###################################################################################################
        denoised_img = subtract_noise_rolling_ball(self.microscope_img, rolling_radius)
        # denoised_img = self.microscope_img
        ###################################################################################################

        # Apply masks
        otsu_mask = apply_otsu(denoised_img)
        dog_mask = apply_dog_triangle(denoised_img, self.gamma)
        procedure_A_mask = otsu_mask | dog_mask

        # Save masks/corrected images
        img_to_file(self.microscope_img, "Original Image.png", self.output_dir)
        img_to_file(denoised_img, "Denoised Image.png", self.output_dir)
        img_to_file(255 - 255 * otsu_mask, "Otsu Mask.png", self.output_dir)
        img_to_file(255 - 255 * dog_mask, "DoG Mask.png", self.output_dir)
        img_to_file(
            255 - 255 * procedure_A_mask, "Procedure A Mask.png", self.output_dir
        )

        # Analyse particle count
        filled_mask = fill_outlines(procedure_A_mask)
        particle_dicts = identify_particles(filled_mask, self.um_per_pixel)

        # Visualise the result
        if self.visualiser_flag:
            show_overlay(denoised_img, procedure_A_mask, particle_dicts)

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

        ###################################################################################################
        # WIP
        ###################################################################################################
        denoised_img = subtract_noise_rolling_ball(self.microscope_img, rolling_radius)
        # denoised_img = self.microscope_img
        ###################################################################################################

        # Apply masks
        otsu_mask = apply_otsu(denoised_img)
        dog_mask = apply_dog_triangle(denoised_img, self.gamma)
        fixed_prominence_mask = apply_fixed_prominence_maxima(denoised_img, prominence)

        procedure_B_mask = (otsu_mask | dog_mask) & fixed_prominence_mask

        # Save masks/images
        img_to_file(self.microscope_img, "Original Image.png", self.output_dir)
        img_to_file(denoised_img, "Denoised Image.png", self.output_dir)
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

        # Visualise the result
        if self.visualiser_flag:
            show_overlay(denoised_img, procedure_B_mask, particle_dicts)

    def procedure_C(self):

        ###################################################################################################
        # WIP
        ###################################################################################################
        # denoised_img = subtract_noise_rolling_ball(self.microscope_img, rolling_radius)
        denoised_img = self.microscope_img
        ###################################################################################################

        # Adjust luminance
        corrected_image = correct_illumination(denoised_img)

        # Apply masks
        otsu_mask = apply_otsu(corrected_image)
        dog_mask = apply_dog_triangle(corrected_image, self.gamma)
        procedure_C_mask = otsu_mask | dog_mask

        # Save masks/images
        img_to_file(self.microscope_img, "Original Image.png", self.output_dir)
        img_to_file(corrected_image, "Corrected Image.png", self.output_dir)
        img_to_file(255 - 255 * otsu_mask, "Otsu Mask.png", self.output_dir)
        img_to_file(255 - 255 * dog_mask, "DoG Mask.png", self.output_dir)
        img_to_file(
            255 - 255 * procedure_C_mask, "Procedure C Mask.png", self.output_dir
        )

        # Analyse particle count
        filled_mask = fill_outlines(procedure_C_mask)
        particle_dicts = identify_particles(filled_mask, self.um_per_pixel)

        # Visualise the result
        if self.visualiser_flag:
            show_overlay(self.microscope_img, procedure_C_mask, particle_dicts)
