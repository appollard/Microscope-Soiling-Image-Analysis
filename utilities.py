from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
import os
import argparse

""" Define the file suffixes that we allow

    Self-explanatory
"""
IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}

""" Define the presets the user can optionally choose from

    Presets A and C replicate the procedures used in the round-robin.
"""

PRESETS = {
    "A": {
        "gamma": 0.9,
        "rolling_ball": True,
        "sliding": True,
        "exclude_edges": True,
        "scale": 1 / 3.156,
        "background": "black",
        "rolling_radius": 50,
        "sigma1": 1,
        "sigma2": 2,
        "DoG_mask": "Triangle",
    },
    "C": {
        "gamma": 1,
        "colour": True,
        "retinex": True,
        "rolling_ball": True,
        "presmoothing": True,
        "sliding": True,
        "close_particles": True,
        "fill_particles": True,
        "de_agglomerate_particles": True,
        "high_precision_DoG": True,
        "exclude_edges": True,
        "scale": 1 / 0.464,
        "retinex_type": "Uniform",
        "retinex_scale": 240,
        "retinex_scale_division": 3,
        "retinex_dynamic": 2.12,
        "background": "white",
        "rolling_radius": 50,
        "sigma1": 1,
        "sigma2": 2,
        "DoG_mask": "Default",
    },
}

BASE_DEFAULTS = {
    "procedure": None,
    "output_dir": None,
    "gamma": 1,
    "colour": False,
    "retinex": False,
    "rolling_ball": False,
    "presmoothing": False,
    "sliding": False,
    "close_particles": False,
    "fill_particles": False,
    "de_agglomerate_particles": False,
    "high_precision_DoG": False,
    "exclude_edges": False,
    "show_visualiser": False,
    "give_diameters": False,
    "fiji_dir": None,
    "input_file": None,
    "input_dir": None,
    "scale": None,
    "retinex_type": None,
    "retinex_scale": None,
    "retinex_scale_division": None,
    "retinex_dynamic": None,
    "background": None,
    "rolling_radius": None,
    "sigma1": 1,
    "sigma2": 2,
    "DoG_mask": None,
}


def get_CLI_args():
    """Parse Command line arguments.

    Returns a namespace item containing CLIs

    """

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    #### PRESETS ####
    parser.add_argument("--procedure", choices=["A", "C"])

    #### RAW VARIABLES ####
    # Misc
    parser.add_argument("--fiji-dir", type=str, required=True)

    # Setup
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
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
    parser.add_argument("--give-diameters", action="store_true")

    return parser.parse_args()


def get_args():
    """Obtain the command line arguments and handle presets

    Start with the default arguments. If a preset is specified, update the arguments with the
    preset. Then update the result with the CLI arguments. This enforces priority between different
    argument sources.
    """
    args = get_CLI_args()

    preset = (
        PRESETS.get(args.procedure, None) if hasattr(args, "procedure") else None
    )  # Extract the desired preset (default to None)

    # Start with the defaults
    final = dict(BASE_DEFAULTS)

    # Update with the preset
    if preset is not None:
        final.update(preset)

    # Update with user arguments
    final.update(vars(args))

    # Convert to namespace
    final = argparse.Namespace(**final)

    return final


def get_files(dir):
    """Returns a list of file names in a directory

    Used to obtain the name of each image to be processed when a specific file name is not
    provided by the user.

    """

    return [
        f.name
        for f in Path(dir).iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES
    ]


def convert_numpy(obj):
    """Convert NumPy objects to native Python

    Lets us save things to jsons.
    """

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
    """Save a list of dicts to a json in a given directory

    Converts a list of dictionaries with numpy elements to native Python. Then saves the resulting
    dictionary to the specified output directory with the specified name.
    """

    # Full path to the output file
    output_path = Path(output_dir) / name

    # Convert non-native types to native Python
    cleaned_dicts = convert_numpy(list_of_dicts)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_dicts, f, indent=1)


def particles_from_imagej(rt, labels, scale):
    """Construct a list of particles and their traits

    Extracts properties of each particle from the results table produced by ImageJ's particle
    analyzer. Computes some additional properties based on this information.

    Returns a list of dictionaries; each dictionary contains the properties of a single particle.

    """

    n = rt.size()

    # Define a function to retrieve entire column from the results table
    def col(name):
        return np.asarray(rt.getColumnAsDoubles(rt.getColumnIndex(name)))

    # Extract info for all aprticles
    area_px = col("Area")
    perimeter_px = col("Perim.")
    bx = col("BX")
    by = col("BY")
    w = col("Width")
    h = col("Height")
    centroid_x = col("X")
    centroid_y = col("Y")

    # Ellipse axes come back in pixels because the image scale is pixels
    major_px = col("Major")
    minor_px = col("Minor")
    major_um = major_px * scale  # Major/minor axes are full, not semi.
    minor_um = minor_px * scale
    area_um2 = area_px * scale**2
    ratio = np.divide(
        minor_px, major_px, out=np.ones_like(minor_px), where=major_px > 0
    )
    eccentricity = np.sqrt(1 - ratio**2)
    diameter = np.sqrt(4 * area_um2 / np.pi)
    volume = 4 / 3 * np.pi * (minor_um / 2) ** 2 * (major_um / 2)

    particle_dicts = []
    for i in range(n):

        particle_dicts.append(
            {
                "label": i + 1,  # Labels start from 1, indexing starts from 0
                "area_px": area_px[i],
                "area_um2": area_um2[i],
                "effective_diameter_um": diameter[i],
                "centroid_px": (centroid_y[i], centroid_x[i]),
                "perimeter_px": perimeter_px[i],
                "eccentricity": eccentricity[i],
                "major_axis_length_um": major_um[i],
                "minor_axis_length_um": minor_um[i],
                "spheroid_volume_um3": volume[i],
                "bbox_px": (by[i], bx[i], by[i] + h[i], bx[i] + w[i]),
            }
        )

    return particle_dicts


def soiling_info(particle_dicts, dimensions, um_per_px):
    """Provides a basic estimation of area and volume soiling.

    Takes a list of dictionaries (one for each particle) and the dimension and scale of the image.
    Area is basic coverage per particle. This is NOT related to reflectance lost; second-surface
    and homogeneity/non-homogeneity assumptions, scattering etc. are all downstream. Any actual
    analysis should replace this function.

    Prints the volume in cubic microns and area in square microns, as well as proportional coverage
    and volume (cubic microns) per square metre of reflector to a json. Note that proportional
    coverage is only meaningful if edge-particles are NOT omitted (since omitting them means that
    covered area is flagged as being clear).
    """

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


def get_histogram(diameters, args):

    fig = Figure(figsize=(6, 4), dpi=200)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(diameters, bins=100)
    ax.set_xlabel(r"Effective diameter ($\mu$m)")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of Effective Particle Diameters")
    fig.savefig(os.path.join(args.output_dir, "Effective Diameters Histogram.png"))


def colourful_particle_map(labels, alpha):
    """Generate a map of particles with different colours.

    Produces a map that's used to visualise the particles. The arguments are the labelled particle
    mask and transparency of particles. Returns an RGBA image with the same 2d dimensions as the input.
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

    # Set the alpha
    map[mask, 3] = int(round(alpha * 255))

    return map


def show_overlay(microscope_img, labels, args):
    """Generate an interactive overlay of the particle map over an image.

    Takes an image (the original microscope image is used normally) and a labelled particle mask.
    Uses colourful_particle_map to turn the laelled particle mask into an RGBA image and overlays
    this on the image input. Produces an interactive figure which lets the user change the alpha
    values (transparency) of the particles, and zoom in/out. Re-renders everything with every
    change, so can get slow for large images.
    """

    # Generate the map using the particle list. Set the alpha of each particle to 0.4
    map = colourful_particle_map(labels, 0.4)

    # Turn this into a picture to be saved to the output directory
    background = Image.fromarray(microscope_img)
    particle_map = Image.fromarray(map)
    background = background.convert("RGB")
    background.paste(particle_map, (0, 0), mask=particle_map)
    background.save(os.path.join(args.output_dir, "Particle Map.png"))

    # Check for the show-visualiser flag
    if args.show_visualiser:

        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)

        # Show the original image
        ax.imshow(microscope_img, cmap="gray", vmin=0, vmax=255)  # Assumes uint8

        # Show the map
        map_overlay = ax.imshow(map)

        # Slider to control map opacity
        ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
        slider = Slider(ax_slider, "Mask opacity", 0.0, 1.0, valinit=0.4)

        def update(val):
            map_overlay.set_alpha(slider.val)
            fig.canvas.draw_idle()

        slider.on_changed(update)
        plt.show()
        plt.close("all")
