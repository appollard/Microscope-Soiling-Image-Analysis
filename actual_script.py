# Package
from image_processing_procedures import SoilingAnalysis

# Run Procedure A
SoilingAnalysis(
    "01.bmp", "black", 1 / 3.156, True, 0.9, "images", "Output Files"
).procedure_A()

# Run Procedure B
SoilingAnalysis(
    "08.bmp", "black", 1 / 3.156, True, 0.9, "images", "Output Files"
).procedure_B(15, 50)
