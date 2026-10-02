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


# Import modules ##################################################################################
import os
import utilities as ut
from scyjava import jimport
from pathlib import Path
import numpy as np

###################################################################################################


# Define each stage ###############################################################################
class PreprocessingStage:
    """Stage that takes Java ij.IJ class, command line arguments and a raw image and conducts
        image preprocessing.

    Methods:
        __init__:
            Assigns the Java class and command line arguments to the self entity so that they may
            be accessed by run().
        run:
            Takes the self and a microscope image as inputs. The microcope image is either a 2d
            array of 8-bit greyscale values or a 3-channel 2d array of 8-bit RGB values. Performs
            rolling ball smoothing if and as specified by the user. Does retinex colour correction
            and converts to greyscale if the image is colour and this is specified by the user.

            Returns the preprocessed 8-bit greyscale image.
    """

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
    """Stage that takes Java ij.IJ class, command line arguments, preprocessed image and produces
    an otsu mask.

    Methods:
        __init__:
            Assigns the Java class and command line arguments to the self entity so that they
            may be accessed by run().
        run:
            Takes the self and a processed 8-bit greyscale image as arguments. Applies the
            otsu masking technique to the image. Adjusts for the background (brightfield or
            darkfield) as specified by the user. Optionally uses ImageJ ring-closing, hole
            filling and/or watersheding as specified by the user.

            Returns a boolean mask of same dimensions as the input image.

            Uploads an image of the mask returned to the output directory.
    """

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
    """Stage that takes Java ij.IJ class, command line arguments, preprocessed image and produces
    a difference-of-gaussians (DoG) mask.

    Methods:
        __init__:
            Assigns the Java class and command line arguments to the self entity so that they
            may be accessed by run().
        run:
            Takes the self and a processed 8-bit greyscale image as arguments. Produces two
            gaussian blurs of the image using standard deviations specified by the user.
            Subtracts the greater SD blur from the lesser. The user specifies whether negative
            values are truncated (sets pixels far from edges to zero). Applies user-specified
            masking technique given user-specified background to obtain a mask.

            Returns a boolean mask of same dimensions as the input image.

            Uploads an image of the mask returned to the output directory.
    """

    def __init__(self, IJ, args):
        self.IJ = IJ
        self.args = args

    def run(self, processed_img):
        img = processed_img.duplicate()
        g1 = img.duplicate()
        g2 = img.duplicate()
        img.close()

        # Ensure that self.args.sigma1 is smaller than self.args.sigma2
        if self.args.sigma1 > self.args.sigma2:
            self.args.sigma1, self.args.sigma2 = self.args.sigma2, self.args.sigma1

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


###################################################################################################
class CombinationStage:
    """Stage that takes Java ij.IJ class, command line arguments, masks and produces combined
    mask.

    Methods:
        __init__:
            Assigns the Java class and command line arguments to the self entity so that they
            may be accessed by run().
        run:
            Takes the self and as many masks as the user wishes as arguments. Applies bitwise
            OR to all masks.

            Returns a boolean mask of same dimensions as the input masks.

            Uploads an image of the mask returned to the output directory.
    """

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


###################################################################################################
class ParticleInfo:
    """Stage that takes Java ij.IJ class, command line arguments, ImageJ gateway and uses ImageJ-
       native particle analysis functions where possible, otherwise native Python, to identify
       and store particle information.

    Methods:
        __init__:
            Assigns the Java class, command line arguments and ImageJ gateway to the self
            entity so that they may be accessed by run().
        run:
            Runs the ImageJ particle analyser and harvests information from its results table.
            Constructs a list where each entry is the dictionary of properties for a given
            particle. Exports this to a json. Uses the particle information to estimate volume and
            area of dust coverage. Exports this to a separate json.

            Returns the list of particle dictionaries and the labelled image (not a boolean mask).
    """

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

        # Label image output- instead of boolean mask, each particle's pixels now contain the label of that particular particle.
        analyzer_options = ParticleAnalyzer.SHOW_ROI_MASKS
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

        # Convert mask to array- pixel value is either 0 (background) or the label of that particle.
        # 1 for the first particle, etc. etc.
        label_imp = analyzer.getOutputImage()
        labels = np.asarray(self.gateway.py.from_java(label_imp)).astype(np.int64)

        # Identify particles from the mask using ImageJ for all particle analysis and Python for simple numerical calculations
        particle_dicts = ut.particles_from_imagej(
            rt,
            labels,
            self.args.scale,
        )

        # Alignment check: table rows and label image must agree
        assert (
            rt.size() == len(particle_dicts) == labels.max()
        ), f"Mismatch: table={rt.size()}, labelled={len(particle_dicts)}, max label={labels.max()}"

        rt.reset()

        # Save particle information to json
        ut.save_to_json(particle_dicts, self.args.output_dir, "Particle_info.json")

        # Extract whole-particle information
        soiling_data = ut.soiling_info(particle_dicts, labels.shape, self.args.scale)

        # Save whole-image information to json
        ut.save_to_json(soiling_data, self.args.output_dir, "Soiling_info.json")

        return particle_dicts, labels


class Visualisation:
    """Stage that takes the labelled image and visualises it against the original image.

    Methods:
         __init__:
             Assigns the Java class, command line arguments and ImageJ gateway to the self
             entity so that they may be accessed by run().
         run:
             Shows an overlay of the particles identified over the original image
    """

    def __init__(self, args, gateway):
        self.args = args
        self.gateway = gateway

    def run(self, original_image, labels):

        # Show visualiser
        if self.args.show_visualiser:
            original_np = self.gateway.py.from_java(original_image)
            ut.show_overlay(original_np, labels)


def main():
    import imagej

    args = ut.get_args()

    ij = imagej.init(args.fiji_dir, mode="headless")
    IJ = jimport("ij.IJ")

    preprocessing = PreprocessingStage(IJ, args)
    otsu = OtsuStage(IJ, args)
    DoG = DoGStage(IJ, args)
    combination = CombinationStage(IJ, args)
    particle_info = ParticleInfo(IJ, args, ij)
    visualiser = Visualisation(args, ij)

    # If no input file provided, iterate through entire folder
    if args.input_file:
        image_files = [args.input_file]
    else:
        image_files = ut.get_files(args.input_dir)

    # Save initial output directory
    initial_output_dir = args.output_dir

    for filename in image_files:

        # If no corresponding output folder, make one
        output_dir = Path(initial_output_dir) / Path(filename).stem
        output_dir.mkdir(parents=True, exist_ok=True)

        # Make it the new output directory
        args.output_dir = str(output_dir)

        microscope_img = IJ.openImage(os.path.join(args.input_dir, filename))

        processed_img = preprocessing.run(microscope_img)
        otsu_mask = otsu.run(processed_img)
        DoG_mask = DoG.run(processed_img)
        combined_mask = combination.run([otsu_mask, DoG_mask])
        particle_dicts, labelled_particles = particle_info.run(combined_mask)
        visualiser.run(microscope_img, labelled_particles)

        # Close images
        processed_img.close()
        combined_mask.close()
        microscope_img.close()

    ij.dispose()
    os._exit(0)  # Need the hard stuff to kill


if __name__ == "__main__":
    main()
