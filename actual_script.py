# Package
from image_processing_procedures import SoilingAnalysis

# Run Procedure A
SoilingAnalysis(
    "09.jpg", "white", 1 / 3.156, True, 0.9, "microscope_images", "outputs"
).procedure_C()

# Run Procedure B
SoilingAnalysis(
    "09.bmp", "black", 1 / 3.156, True, 0.9, "microscope_images", "outputs"
).procedure_B(15, 50)
