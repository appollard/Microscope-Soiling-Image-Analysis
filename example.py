import os
from pathlib import Path
import utilities as ut
from scyjava import jimport
from skimage.measure import label, regionprops

import argparse


def get_args():
    parser = argparse.ArgumentParser()

    # Misc
    parser.add_argument("--fiji-dir", default=r"C:\Users\snare\Fiji.app")

    # Setup
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--input_dir", default="microscope_images")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--scale", type=float, default=1 / 3.156)

    # Preprocessing
    parser.add_argument("--colour", action="store_true")
    parser.add_argument("--retinex", action="store_true")
    parser.add_argument(
        "--retinex-type", choices=["Uniform", "Low", "High"], default="Uniform"
    )
    parser.add_argument("--retinex-scale", type=int, default=240)
    parser.add_argument("--retinex-scale-division", type=int, default=3)
    parser.add_argument("--retinex-dynamic", type=float, default=2.12)

    parser.add_argument("--background", choices=["white", "black"], default="white")
    parser.add_argument("--rolling-ball", action="store_true")
    parser.add_argument("--presmoothing", action="store_true")
    parser.add_argument("--sliding", action="store_true")
    parser.add_argument("--rolling-radius", type=int, default=50)

    # Mask 1: Otsu
    parser.add_argument("--close-particles", action="store_true")
    parser.add_argument("--fill-particles", action="store_true")
    parser.add_argument("--de-agglomerate-particles", action="store_true")

    # Mask 2: DoG
    parser.add_argument("--sigma1", type=float, default=1.0)
    parser.add_argument("--sigma2", type=float, default=2.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument(
        "--DoG-mask", choices=["Triangle", "Default"], default="Triangle"
    )  # need to check background colour when processing

    return parser.parse_args()


class PreprocessingStage:
    def __init__(self, IJ, args):
        self.IJ = IJ
        self.args = args

    def run(self, microscope_img):
        img = microscope_img.duplicate()

        # Set scale
        self.IJ.run(
            img, "Set Scale...", f"distance={self.args.scale} known=1 unit=mu.m"
        )

        # Setup rolling ball smoothing string
        if self.args.rolling_ball:
            flags = [f"rolling={self.args.rolling_radius}"]

            if self.args.background == "white":
                flags.append("light")
            if self.args.sliding:
                flags.append("sliding")
            if not self.args.presmoothing:
                flags.append("disable")

            rolling_ball_str = " ".join(flags)

        # Setup retinex string
        if self.args.retinex:
            flags = [f"level={self.args.retinex_type}"]
            flags.append(f"scale={self.args.retinex_scale}")
            flags.append(f"scale_division={self.args.retinex_scale_division}")
            flags.append(f"dynamic={self.args.retinex_dynamic}")

            retinex_str = " ".join(flags)

        # Handle coloured images
        if self.args.colour:

            # Handle rolling ball smoothing
            if self.args.rolling_ball:
                self.IJ.run(img, "Subtract Background...", rolling_ball_str)

            # Handle retinex
            if self.args.retinex:
                self.IJ.run(img, "Retinex", retinex_str)

            # Convert to 8-bit greyscale
            self.IJ.run(img, "8-bit", "")

        else:

            # Convert to 8-bit greyscale
            self.IJ.run(img, "8-bit", "")

            # Handle rolling ball smoothing
            if self.args.rolling_ball:
                self.IJ.run(img, "Subtract Background...", rolling_ball_str)

        return img


class OtsuStage:
    def __init__(self, IJ, args):
        self.IJ = IJ
        self.args = args

    def run(self, processed_img):
        img = processed_img.duplicate()

        # Setup otsu string
        flags = ["Otsu"]
        if self.args.background == "black":
            flags.append("dark")
        otsu_str = " ".join(flags)

        # Get 8-bit otsu mask
        self.IJ.setAutoThreshold(img, otsu_str)

        # Convert to boolean mask
        self.IJ.run(img, "Convert to Mask", "")

        # Optionally run ring-closing
        if self.args.close_particles:
            self.IJ.run(img, "Close-", "")

        # Optionally run particle filling
        if self.args.fill_particles:
            self.IJ.run(img, "Fill Holes", "")

        # Optionally run watershedding
        if self.args.de_agglomerate_particles:
            self.IJ.run(img, "Watershed", "")

        # Save mask image
        self.IJ.saveAs(img, "png", os.path.join(self.args.output_dir, "Otsu Mask"))

        return img


class DoGStage:
    def __init__(self, IJ, args):
        self.IJ = IJ
        self.args = args

    def run(self, processed_img):
        img = processed_img.duplicate()
        g1 = img.duplicate()
        g2 = img.duplicate()

        # Obtain the blurs
        self.IJ.run(g1, "Gaussian Blur...", f"sigma={self.args.sigma1}")
        self.IJ.run(g2, "Gaussian Blur...", f"sigma={self.args.sigma2}")

        # Obtain image calculator
        ic = jimport("ij.plugin.ImageCalculator")()

        # Calculate the DoG
        img = ic.run("Subtract create 32-bit", g1, g2)

        # Apply gamma correction (default of 1, does nothing)
        self.IJ.run(img, "Gamma...", f"value={self.args.gamma}")

        # Set float32_t -> uint8_t conversion to rescale rather than truncate
        self.IJ.run(img, "Conversions...", "scale")

        # Convert to greyscale
        self.IJ.run(img, "8-bit", "")

        # Setup threshold string
        flags = [self.args.DoG_mask]
        if self.args.background == "black":
            flags.append("dark")
        DoG_str = " ".join(flags)

        # Get 8-bit DoG mask
        self.IJ.setAutoThreshold(img, DoG_str)

        # Convert to boolean mask
        self.IJ.run(img, "Convert to Mask", "")

        # Cleanup
        for img in [g1, g2]:
            if img is not None:
                img.close()

        # Save mask image
        self.IJ.saveAs(img, "png", os.path.join(self.args.output_dir, "DoG Mask"))

        return img


class CombinationStage:
    def __init__(self, IJ, args):
        self.IJ = IJ
        self.args = args

    def run(self, masks):

        # Obtain image calculator
        ic = jimport("ij.plugin.ImageCalculator")()

        # Conduct bitwise OR of all input masks (have to do two at a time)
        mask = masks[0]
        for m in masks[1:]:
            mask = ic.run("OR create", mask, m)

        # Clean up remaining images
        for item in masks:
            if item is not None:
                item.close()

        # Save mask image
        self.IJ.saveAs(mask, "png", os.path.join(self.args.output_dir, "Final Mask"))

        return mask


class ParticleInfo:
    def __init__(self, IJ, args, gateway):
        self.IJ = IJ
        self.args = args
        self.gateway = gateway

    def run(self, mask):

        # Convert mask to numpy array
        mask_np = self.gateway.py.from_java(mask)

        # Identify particles and retrieve information
        particle_dicts = ut.identify_particles(mask_np, self.args.scale)

        # Save particle information to json
        ut.save_to_json(particle_dicts, self.args.output_dir, "Particle_info.json")

        return particle_dicts


def main():
    import imagej

    args = get_args()

    ij = imagej.init(args.fiji_dir, mode="headless")
    IJ = jimport("ij.IJ")

    microscope_img = IJ.openImage(os.path.join(args.input_dir, args.input_file))

    preprocessing = PreprocessingStage(IJ, args)
    otsu = OtsuStage(IJ, args)
    DoG = DoGStage(IJ, args)
    combination = CombinationStage(IJ, args)
    particle_info = ParticleInfo(IJ, args, ij)

    processed_img = preprocessing.run(microscope_img)
    otsu_mask = otsu.run(processed_img)
    DoG_mask = DoG.run(processed_img)
    combined_mask = combination.run([otsu_mask, DoG_mask])
    particle_dicts = particle_info.run(combined_mask)


if __name__ == "__main__":
    main()

# Procedure A
