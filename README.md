# COCO Person Segmentation and Patch Extraction Pipeline

## Overview

This project processes images from the COCO dataset to isolate and extract human subjects using segmentation masks. The pipeline generates:

- person outlines,
- surrounding border patches,
- cropped person-centered images,
- padded outputs for downstream machine learning tasks.

The script leverages COCO annotations to create masked and cropped representations of people while removing the background.

---

# Features

The pipeline performs the following operations:

- Loads COCO person annotations
- Extracts segmentation masks
- Draws contours around people
- Generates surrounding patches around the segmentation boundary
- Removes background pixels
- Crops images tightly around detected people
- Pads cropped outputs with configurable margins
- Saves both cropped and non-cropped processed images

---

# Dataset

This project uses the **COCO 2017 Dataset**.

Required files:

- `annotations/instances_train2017.json`
- COCO training images

---

# Project Structure

```text
project/
│
├── saved_images/
│   ├── 000000000036.jpg
│   └── ...
│
├── annotations/
│   └── instances_train2017.json
│
├── result_non_cropped/
│
├── result_cropped/
│
├── result_images/
│
└── main_script.py
```

---

# Dependencies

Install required packages:

```bash
pip install numpy opencv-python matplotlib pillow tqdm pycocotools requests
```

---

# Core Pipeline

## 1. Load COCO Dataset

The script initializes the COCO API and loads all images containing the `person` category.

---

## 2. Extract Person Masks

For each image:
- all person masks are combined,
- non-person pixels are set to black.

## 3. Draw Person Outlines

Contours are drawn around segmentation polygons using OpenCV.

This creates a visible outline around detected people.

---

## 4. Generate Surrounding Patch

The outline is expanded outward using a configurable thickness parameter.

This creates a border region surrounding the person.

---

## 5. Overlay Original Pixels

Only pixels inside the generated patch retain their original RGB values.

All other pixels are set to black.

---

## 6. Crop to Bounding Rectangle

The script removes unnecessary black regions by locating:
- top-most non-black row,
- bottom-most non-black row,
- left-most non-black column,
- right-most non-black column.

---

## 7. Pad Final Output

The cropped image is padded with black pixels to provide spacing around the subject.

---

# Example Pipeline Output

The visualization pipeline generates:

1. Original Image
2. Person Outline
3. Thick Border Mask
4. Overlay Patch
5. Cropped Output
6. Final Padded Image

---

# Saving Results

Processed outputs are automatically saved in cropped and noncropped folders.

---
---

# Application

This preprocessing pipeline was used to improve a Computer Vision Model for the Rutgers Rail Team Research Group that 
partners with NJ Transit. A paper to the work published after incorporating this is attached here: 
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5208189

---
