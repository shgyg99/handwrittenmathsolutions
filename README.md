
# Persian Handwritten Math Solutions Dataset with Formula Annotations

This dataset contains a collection of images of handwritten math solutions in Persian. Alongside each image, there are JSON files that specify the coordinates of the formulas present in the images.

## Dataset Details
- **Images**: Contain pictures of handwritten math solutions in Persian.
- **JSON Files**: Each JSON file specifies the coordinates of formulas, numbers, and text present in the images. This information can be used for formula recognition and related tasks.

## Dataset Structure
```
/dataset
  /images
    - im (1).jpg
    - im (2).jpg
    - ...
  /annotation
    - im (1).json
    - im (2).json
    - ...
```

- The `images` folder contains image files that show the handwritten math solutions.
- The `annotation` folder contains JSON files that store the coordinates and information of formulas in each image.

## Use Cases
This dataset can be used in various projects such as:
- **Mathematical formula recognition** in handwritten images
- **Image-to-math conversion** (OCR for formulas)
- **Training machine learning models** for formula and equation recognition and analysis

## How to Use
To use this dataset, you can download the image and JSON files and use them in your various projects. If you need further processing, you can extract the coordinates of the formulas from the JSON files and use them for training NLP models or formula detection.
