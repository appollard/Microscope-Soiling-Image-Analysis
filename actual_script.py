# Package
import imagej
from image_processing_procedures import SoilingAnalysis

ij = imagej.init(r"C:\Users\snare\Fiji.app", mode="headless")

SoilingAnalysis(
    "01.bmp", "black", ij, 1 / 3.156, True, 0.9, "microscope_images", "outputs"
).procedure_A()

SoilingAnalysis(
    "09.jpg", "white", 1 / 3.156, True, 0.9, "microscope_images", "outputs"
).procedure_C([2, 82, 162], 50)
