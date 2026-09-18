import os
import sys
import utilities as ut
from scyjava import jimport


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
        img.close()

        # Obtain the blurs
        self.IJ.run(g1, "Gaussian Blur...", f"sigma={self.args.sigma1}")
        self.IJ.run(g2, "Gaussian Blur...", f"sigma={self.args.sigma2}")

        # Obtain image calculator
        ic = jimport("ij.plugin.ImageCalculator")()

        # Calculate the DoG using high or not-high precision
        if self.args.high_precision_DoG:
            img = ic.run("Subtract create 32-bit", g1, g2)
            self.IJ.run(img, "Conversions...", "scale")
            self.IJ.run(img, "8-bit", "")
        else:
            img = ic.run("Subtract create", g1, g2)

        # Set float32_t -> uint8_t conversion to rescale rather than truncate
        self.IJ.run(img, "Conversions...", "scale")

        # Apply gamma correction (default of 1, does nothing)
        self.IJ.run(img, "Gamma...", f"value={self.args.gamma}")

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
        for item in [g1, g2]:
            if item is not None:
                item.close()

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

        ResultsTable = jimport("ij.measure.ResultsTable")
        Measurements = jimport("ij.measure.Measurements")
        ParticleAnalyzer = jimport("ij.plugin.filter.ParticleAnalyzer")

        # Make sure all values are in pixels so we can do the processing ourselves
        self.IJ.run(mask, "Set Scale...", "distance=0 known=0 unit=pixel")

        # Use the analyzer API directly so headless ImageJ does not open its dialog.
        rt = ResultsTable.getResultsTable()
        rt.reset()

        # Define the measurements we will receive from ImageJ's analyzer
        measurements = (
            Measurements.AREA
            | Measurements.CENTROID
            | Measurements.PERIMETER
            | Measurements.RECT
            | Measurements.ELLIPSE
            | Measurements.SHAPE_DESCRIPTORS
        )

        analyzer_options = ParticleAnalyzer.CLEAR_WORKSHEET | ParticleAnalyzer.SHOW_NONE
        if self.args.exclude_edges:
            analyzer_options |= ParticleAnalyzer.EXCLUDE_EDGE_PARTICLES

        # Call the particle analyzer
        analyzer = ParticleAnalyzer(
            analyzer_options,
            measurements,
            rt,
            0.0,
            float("inf"),
        )
        analyzer.analyze(mask)

        # Convert mask to numpy array
        mask_np = self.gateway.py.from_java(mask)

        # Identify particles from the mask because ROI Manager is GUI-only in headless mode.
        particle_dicts = ut.identify_particles_from_mask(
            rt,
            mask_np,
            self.args.scale,
            self.args.exclude_edges,
        )

        rt.reset()

        # Save particle information to json
        ut.save_to_json(particle_dicts, self.args.output_dir, "Particle_info.json")

        # Extract whole-particle information
        soiling_data = ut.soiling_info(particle_dicts)

        # Save whole-image information to json
        ut.save_to_json(soiling_data, self.args.output_dir, "Soiling_info.json")

        return particle_dicts


class Visualisation:
    def __init__(self, args, gateway):
        self.args = args
        self.gateway = gateway

    def run(self, particle_dicts, original_image, mask):

        # Show visualiser
        if self.args.show_visualiser:
            original_np = self.gateway.py.from_java(original_image)
            mask_np = self.gateway.py.from_java(mask)
            ut.show_overlay(original_np, mask_np, particle_dicts)


def main():
    import imagej

    args = ut.get_args()

    ij = imagej.init(args.fiji_dir, mode="headless")
    IJ = jimport("ij.IJ")

    microscope_img = IJ.openImage(os.path.join(args.input_dir, args.input_file))

    preprocessing = PreprocessingStage(IJ, args)
    otsu = OtsuStage(IJ, args)
    DoG = DoGStage(IJ, args)
    combination = CombinationStage(IJ, args)
    particle_info = ParticleInfo(IJ, args, ij)
    visualiser = Visualisation(args, ij)

    processed_img = preprocessing.run(microscope_img)
    otsu_mask = otsu.run(processed_img)
    DoG_mask = DoG.run(processed_img)
    combined_mask = combination.run([otsu_mask, DoG_mask])
    particle_dicts = particle_info.run(combined_mask)
    visual_interface = visualiser.run(particle_dicts, microscope_img, combined_mask)

    ij.dispose()
    sys.exit(0)
    return 0


if __name__ == "__main__":
    main()

# Command line structure. Entire commands inside square brackets indicate that
# this is set to true if the command is included:

# --fiji-dir [YOUR DIRECTORY FOR Fiji.app]  --input-file [IMAGE NAME] --input-dir [FOLDER IT'S IN] --output-dir [FOLDER TO SAVE THE OUTPUTS]
# --scale [PIXEL SCALE] [--colour] [--retinex] --retinex-type [TYPE] --retinex-scale [SCALE] --retinex-scale-division [DIVS]
# --retinex-dynamic [DYNAMIC] --background [COLOUR] [--rolling-ball] [--presmoothing] [--sliding] --rolling-radius [RADIUS]
# [--close-particles] [--fill-particles] [--de-agglomerate-particles] [--high-precision-DoG] --sigma1 [VAL] --sigma2 [VAL] --gamma [VAL] --DoG-mask [MASK TYPE]
# [--exclude-edges] [--show-visualiser]


# Procedure A Copy-paste for 01.bmp image (fill in your info):

# --fiji-dir "[Fiji.app DIRECTORY]" --input-file "01.bmp" --input-dir "[ROOT DIRECTORY INPUT FOLDER]" --output-dir "[ROOT DIRECTORY OUTPUT FOLDER]" --scale 0.31685 --background "black" --rolling-ball --sliding --rolling-radius 50 --sigma1 1 --sigma2 2 --gamma 0.9 --DoG-mask "Triangle" --exclude-edges --show-visualiser

# Procedure A Copy-paste examples (with and without preset):

# --fiji-dir "C:\Users\snare\Fiji.app" --input-file "01.bmp" --input-dir "microscope_images" --output-dir "outputs" --show-visualiser --procedure A
# --fiji-dir "C:\Users\snare\Fiji.app" --input-file "01.bmp" --input-dir "microscope_images" --output-dir "outputs" --scale 0.31685 --background "black" --rolling-ball --sliding --rolling-radius 50 --sigma1 1 --sigma2 2 --gamma 0.9 --DoG-mask "Triangle" --exclude-edges --show-visualiser


# Procedure C Copy-paste for 09.jpg image (fill in your info):

# --fiji-dir [Fiji.app DIRECTORY]  --input-file "09.jpg" --input-dir [ROOT DIRECTORY INPUT FOLDER] --output-dir [ROOT DIRECTORY OUTPUT FOLDER] --scale 2.155 --colour --retinex --retinex-type "Uniform" --retinex-scale 240 --retinex-scale-division 3 --retinex-dynamic 2.12 --background "white" --rolling-ball --presmoothing --sliding --rolling-radius 50 --close-particles --fill-particles --de-agglomerate-particles --high-precision-DoG --sigma1 1 --sigma2 2 --gamma 1 --DoG-mask "Default" --exclude-edges --show-visualiser

# Procedure C Copy-paste example:

# --fiji-dir "C:\Users\snare\Fiji.app"  --input-file "09.jpg" --input-dir "microscope_images" --output-dir "outputs" --scale 2.155  --colour --retinex --retinex-type "Uniform" --retinex-scale 240 --retinex-scale-division 3 --retinex-dynamic 2.12 --background "white" --rolling-ball --presmoothing --sliding --rolling-radius 50 --close-particles --fill-particles --de-agglomerate-particles --high-precision-DoG --sigma1 1 --sigma2 2 --gamma 1 --DoG-mask "Default" --exclude-edges --show-visualiser
