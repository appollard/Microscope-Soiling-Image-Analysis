from pathlib import Path
import json
import numpy as np


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
        json.dump(cleaned_dicts, f, indent=1)


def identify_particles(rt, rois, scale, gateway):
    particle_dicts = []
    for i, roi in enumerate(rois):
        area_px = rt.getValue("Area", i)
        perimeter_px = rt.getValue("Perim.", i)
        centroid_x = rt.getValue("X", i)
        centroid_y = rt.getValue("Y", i)
        bx = rt.getValue("BX", i)
        by = rt.getValue("BY", i)
        width = rt.getValue("Width", i)
        height = rt.getValue("Height", i)
        major = rt.getValue("Major", i)
        minor = rt.getValue("Minor", i)

        eccentricity = np.sqrt(1 - (minor / major) ** 2) if major > 0 else 0.0
        area_um2 = area_px * (scale**2)
        effective_diameter = np.sqrt(4 * area_um2 / np.pi)

        # Pixel coordinates for this particle: crop mask + bounding box offset
        roi_mask_np = gateway.py.from_java(roi.getMask())
        bounds = roi.getBounds()
        rows, cols = np.nonzero(roi_mask_np)
        coords = [(int(r + bounds.y), int(c + bounds.x)) for r, c in zip(rows, cols)]

        particle_dicts.append(
            {
                "label": i + 1,
                "area_px": area_px,
                "area_um2": area_um2,
                "effective_diameter": effective_diameter,
                "centroid": (
                    centroid_y,
                    centroid_x,
                ),  # (row, col), matches regionprops convention
                "perimeter_px": perimeter_px,
                "eccentricity": eccentricity,
                "major_axis_length": major,
                "minor_axis_length": minor,
                "bbox": (
                    by,
                    bx,
                    by + height,
                    bx + width,
                ),  # (min_row, min_col, max_row, max_col)
                "coords": coords,
            }
        )
    return particle_dicts
