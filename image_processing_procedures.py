# Packages
import numpy as np
import os
import cv2
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.ndimage import binary_fill_holes
from skimage.measure import label, regionprops
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


def convert_numpy(obj):
    # Convert numpy scalars
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()

    # Convert numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # Convert dicts
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}

    # Convert lists/tuples
    if isinstance(obj, (list, tuple)):
        return [convert_numpy(v) for v in obj]

    # Everything else stays as-is
    return obj


def save_to_json(list_of_dicts, output_dir, name):

    # Root directory where this script lives
    root_dir = Path(__file__).parent

    # Folder inside the root directory
    save_dir = root_dir / output_dir
    save_dir.mkdir(exist_ok=True)

    # Full path to the output file
    output_path = save_dir / name

    # Convert non-native types to native Python
    cleaned_dicts = convert_numpy(list_of_dicts)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_dicts, f, indent=4)


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
        self.threshold_method = None
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

        # Initialise
        subtracted_background = self.microscope_img.duplicate()
        self.IJ.run(subtracted_background, "8-bit", "")

        # Handle Background
        self.IJ.run(
            subtracted_background,
            "Subtract Background...",
            "rolling=50 sliding disable",
        )
        # self.IJ.run(subtracted_background, "Enhance Contrast", "saturated=0.35")

        # Handle Otsu
        self.threshold_method = "Triangle dark"
        mask1 = subtracted_background.duplicate()
        mask1.setTitle("Mask1")
        if self.background == "black":  # THIS LOGIC IS NOT PRESENT IN THE ORIGINAL
            self.IJ.setAutoThreshold(mask1, "Otsu dark")  # THIS IS HARDCODED
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

        self.IJ.sigmaG1 = 1
        self.IJ.sigmaG2 = 2
        self.IJ.run(g1, "Gaussian Blur...", f"sigma={self.IJ.sigmaG1}")
        self.IJ.run(g2, "Gaussian Blur...", f"sigma={self.IJ.sigmaG2}")

        ic = jimport("ij.plugin.ImageCalculator")()
        mask2 = ic.run("Subtract create", g1, g2)
        self.IJ.run(mask2, "Gamma...", f"value={self.gamma}")
        # self.IJ.run(mask2, "Enhance Contrast", "saturated=0.35")
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

        # Save particle info
        save_to_json(particle_dicts, self.output_dir, "particle_info")

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

    def procedure_B(self):
        """Run Procedure B: apply Otsu, DoG and fixed prominence masks, analyse particles.

        Shelved.
        """

    def procedure_C(self):
        """Run Procedure C: apply Retinex, convert to greyscale,
           Otsu and DoG masks, analyse particles.

        Saves:
            'Otsu Mask.png', 'DoG Mask.png', 'Procedure C Mask.png' to output directory.

        Displays:
            Matplotlib overlay figure if visualiser_flag is True.
        """

        # Handle Background
        subtracted_background = self.microscope_img.duplicate()
        self.IJ.run(
            subtracted_background,
            "Subtract Background...",
            "rolling=50 light sliding",
        )

        # Colour Correction
        self.IJ.run(
            subtracted_background,
            "Retinex",
            "level=Uniform scale=240 scale_division=3 dynamic=2.12",
        )
        self.IJ.run(subtracted_background, "8-bit", "")

        # Handle Otsu
        self.threshold_method = "Default"
        mask1 = subtracted_background.duplicate()
        mask1.setTitle("Mask1")
        self.IJ.setAutoThreshold(mask1, "Otsu")

        self.IJ.run(mask1, "Convert to Mask", "")
        # Extra processing
        self.IJ.run(mask1, "Close-", "")
        self.IJ.run(mask1, "Fill Holes", "")
        self.IJ.run(mask1, "Watershed", "")

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
        mask2 = ic.run("Subtract create 32-bit", g1, g2)
        self.IJ.run(mask2, "Conversions...", "scale")
        self.IJ.run(mask2, "8-bit", "")
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
            subtracted_background,
            "png",
            os.path.join(self.output_dir, "Processed Image"),
        )
        self.IJ.saveAs(
            binary_image, "png", os.path.join(self.output_dir, "Procedure C Mask")
        )

        # Analyse particle count
        mask_np = self.gateway.py.from_java(binary_image)
        filled_mask = fill_outlines(mask_np)
        particle_dicts = identify_particles(filled_mask, self.um_per_pixel)

        # Save particle info
        save_to_json(particle_dicts, self.output_dir, "particle_info")

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
