#!/usr/bin/env python3
"""
Generate a PDF document describing the Attention-Enhanced U-Net Segmentation project.
"""

from fpdf import FPDF


class ProjectPDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 14)
        self.cell(0, 10, 'Attention-Enhanced U-Net Road Segmentation Project', align='C', new_x='LMARGIN', new_y='NEXT')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')


def create_pdf():
    pdf = ProjectPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # GitHub Repository Location
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, 'Code Location:', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Helvetica', '', 11)
    pdf.set_text_color(0, 0, 255)
    pdf.cell(0, 8, 'https://github.com/ajanakiraman2024/attention-enhanced-unet-segmentation', new_x='LMARGIN', new_y='NEXT', link='https://github.com/ajanakiraman2024/attention-enhanced-unet-segmentation')
    pdf.ln(10)

    # Reset text color
    pdf.set_text_color(0, 0, 0)

    # Project Overview
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 10, 'Project Overview', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Helvetica', '', 11)
    pdf.multi_cell(0, 6, 
        'This project implements an Attention-Enhanced U-Net architecture for road segmentation '
        'from satellite imagery. The model uses attention gates to focus on relevant features '
        'and improve segmentation accuracy compared to standard U-Net.')
    pdf.ln(10)

    # Project Structure
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 10, 'Project Structure', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Courier', '', 9)
    
    structure = """attention-enhanced-unet-segmentation/
|
+-- config.py                    # Configuration settings
+-- train.py                     # Training script
+-- evaluate.py                  # Evaluation script
+-- inference.py                 # Inference/prediction script
+-- requirements.txt             # Python dependencies
+-- README.md                    # Project documentation
|
+-- models/                      # Neural network architectures
|   +-- __init__.py
|   +-- unet.py                  # Standard U-Net model
|   +-- attention_unet.py        # Attention U-Net model
|
+-- utils/                       # Utility functions
|   +-- __init__.py
|   +-- dataset.py               # Data loading & preprocessing
|   +-- losses.py                # Loss functions
|   +-- metrics.py               # Evaluation metrics
|
+-- notebooks/                   # Jupyter notebooks
|   +-- UNet_Road_Segmentation_Colab.ipynb
|
+-- data/                        # Dataset
|   +-- training/
|   |   +-- images/              # Original images (100 PNG)
|   |   +-- groundtruth/         # Ground truth masks (100 PNG)
|   |   +-- images_generated/    # Augmented images (1003 PNG)
|   |   +-- groundtruth_generated/  # Augmented masks (1003 PNG)
|   +-- test_set_images/         # Test data (50 images)
|
+-- images/                      # Visualization outputs
+-- outputs/                     # Model checkpoints"""

    pdf.multi_cell(0, 5, structure)

    # Save PDF
    output_path = 'Project_Structure.pdf'
    pdf.output(output_path)
    print(f"PDF generated successfully: {output_path}")
    return output_path


if __name__ == '__main__':
    create_pdf()
