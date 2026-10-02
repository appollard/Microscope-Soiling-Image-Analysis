from pathlib import Path
import json
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import argparse


def get_CLI_args():
    parser = argparse.ArgumentParser()

    #### PRESETS ####
    parser.add_argument("--procedure", choices=["A", "C"])

    #### RAW VARIABLES ####
    # Misc
    parser.add_argument("--fiji-dir", default=r"C:\Users\snare\Fiji.app")

    # Setup
    parser.add_argument("--input-file")
    parser.add_argument("--input-dir", default="microscope_images")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--scale", type=float)

    # Preprocessing
    parser.add_argument("--colour", action="store_true")
    parser.add_argument("--retinex", action="store_true")
    parser.add_argument("--retinex-type", choices=["Uniform", "Low", "High"])
    parser.add_argument("--retinex-scale", type=int)
    parser.add_argument("--retinex-scale-division", type=int)
    parser.add_argument("--retinex-dynamic", type=float)

    parser.add_argument("--background", choices=["white", "black"])
    parser.add_argument("--rolling-ball", action="store_true")
    parser.add_argument("--presmoothing", action="store_true")
    parser.add_argument("--sliding", action="store_true")
    parser.add_argument("--rolling-radius", type=int)

    # Mask 1: Otsu
    parser.add_argument("--close-particles", action="store_true")
    parser.add_argument("--fill-particles", action="store_true")
    parser.add_argument("--de-agglomerate-particles", action="store_true")

    # Mask 2: DoG
    parser.add_argument("--high-precision-DoG", action="store_true")
    parser.add_argument("--sigma1", type=float)
    parser.add_argument("--sigma2", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--DoG-mask", choices=["Triangle", "Default"])

    # Particle analysis
    parser.add_argument("--exclude-edges", action="store_true")
    parser.add_argument("--show-visualiser", action="store_true")

    return parser.parse_args()


def get_args():
    args = get_CLI_args()

    if args.procedure:
        config = PRESETS[args.procedure]

        for key, value in config.items():
            current = getattr(args, key)
            if current is None or current is False:
                setattr(args, key, value)

    return args


PRESETS = {
    "A": {
        "scale": 0.31685,
        "background": "black",
        "rolling_ball": True,
        "sliding": True,
        "rolling_radius": 50,
        "sigma1": 1,
        "sigma2": 2,
        "gamma": 0.9,
        "DoG_mask": "Triangle",
        "exclude_edges": True,
    },
    "C": {
        "input_file": "09.jpg",
        "input_dir": "microscope_images",
        "output_dir": "outputs",
        "scale": 2.155,
        "colour": True,
        "retinex": True,
        "retinex_type": "Uniform",
        "retinex_scale": 240,
        "retinex_scale_division": 3,
        "retinex_dynamic": 2.12,
        "background": "white",
        "rolling_ball": True,
        "presmoothing": True,
        "sliding": True,
        "rolling_radius": 50,
        "close_particles": True,
        "fill_particles": True,
        "de_agglomerate_particles": True,
        "high_precision_DoG": True,
        "sigma1": 1,
        "sigma2": 2,
        "gamma": 1,
        "DoG_mask": "Default",
        "exclude_edges": True,
    },
}


IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def get_files(dir):

    return [
        f.name
        for f in Path(dir).iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES
    ]


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
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Full path to the output file
    output_path = Path(output_dir) / name

    # Convert non-native types to native Python
    cleaned_dicts = convert_numpy(list_of_dicts)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_dicts, f, indent=1)


def particles_from_imagej(rt, labels, scale):
    """Build particle records from an ImageJ results table and its count-mask image."""
    n = rt.size()

    # Group pixel coordinates by label in one pass (avoids n full-image scans)
    flat = labels.ravel()
    order = np.argsort(flat, kind="stable")
    sorted_labels = flat[order]
    starts = np.searchsorted(sorted_labels, np.arange(1, n + 2))
    width = labels.shape[1]

    particle_dicts = []
    for i in range(n):
        label = i + 1
        idx = order[starts[i] : starts[i + 1]]
        coords = [(int(p // width), int(p % width)) for p in idx]

        area_px = rt.getValue("Area", i)
        bx = rt.getValue("BX", i)
        by = rt.getValue("BY", i)
        w = rt.getValue("Width", i)
        h = rt.getValue("Height", i)

        # Ellipse axes come back in pixels because the image scale is pixels
        major_px = rt.getValue("Major", i)
        minor_px = rt.getValue("Minor", i)
        major_um = major_px * scale  # Major/minor axes are full, not semi.
        minor_um = minor_px * scale
        eccentricity = np.sqrt(1 - (minor_px / major_px) ** 2) if major_px > 0 else 0.0
        area_um2 = area_px * scale**2

        particle_dicts.append(
            {
                "label": label,
                "area_px": area_px,
                "area_um2": area_um2,
                "effective_diameter_um": np.sqrt(4 * area_um2 / np.pi),
                "centroid_px": (rt.getValue("Y", i), rt.getValue("X", i)),
                "perimeter_px": rt.getValue("Perim.", i),
                "eccentricity": eccentricity,
                "major_axis_length_um": major_um,
                "minor_axis_length_um": minor_um,
                "spheroid_volume_um3": 4
                / 3
                * np.pi
                * (minor_um / 2) ** 2
                * (major_um / 2),
                "bbox_px": (by, bx, by + h, bx + w),
                "coords": coords,
            }
        )
    return particle_dicts


def soiling_info(particle_dicts, dimensions, um_per_px):
    nrows = dimensions[0]
    ncols = dimensions[1]

    canvas_area_um2 = nrows * ncols * (um_per_px**2)

    soil_volume_um3 = 0
    soil_area_um2 = 0

    for particle in particle_dicts:
        soil_volume_um3 = soil_volume_um3 + particle["spheroid_volume_um3"]
        soil_area_um2 = soil_area_um2 + particle["area_um2"]

    soil_area_coverage = soil_area_um2 / canvas_area_um2
    soil_volume_um3_per_m2 = soil_volume_um3 / canvas_area_um2 * (1e6**2)
    soiling_data = {
        "soil_volume_um3": soil_volume_um3,
        "soil_area_um2": soil_area_um2,
        "soil_area_coverage_decimal": soil_area_coverage,
        "soil_volume_um3_per_m2": soil_volume_um3_per_m2,
    }
    return soiling_data


def colourful_particle_map(labels):
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
    # + (4,) gives the tensor a depth of 4: 3 for RGB and 1 for transparency
    # (default is 0 for transparent canvas, changes to 1 when colour is assigned)
    map = np.zeros((labels.shape + (4,)), dtype=np.uint8)

    # Obtain each pixel which is part of a particle
    mask = labels > 0

    # Set each entry in map which corresponds to a pixel flagged in the mask to a value in colours determined by the value of the particle the pixel belongs to
    map[mask] = colours[labels[mask] % len(colours)]

    return map


# Overlay this on the microscope image for comparison
def show_overlay(microscope_img, labels):
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
    map = colourful_particle_map(labels)

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
