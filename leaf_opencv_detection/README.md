# Leaf OpenCV Detection

This project focuses on detecting and analyzing leaf features using OpenCV, a powerful computer vision library. The goal is to process images of leaves to extract useful information such as shape, size, and texture.

## Features

- Leaf image preprocessing (grayscale conversion, noise reduction, etc.)
- Contour detection and feature extraction
- Shape analysis and classification
- Integration with OpenCV for efficient image processing

## Prerequisites

- Python 3.x
- OpenCV library (`cv2`)
- NumPy

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/theonegareth/Plantesa-training.git
    cd Plantesa-training/leaf_opencv_detection
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Place your leaf images in the `images/` directory.
2. Run the detection script:
    ```bash
    python detect_leaves.py
    ```
3. Processed images and results will be saved in the `output/` directory.

## File Structure

```
leaf_opencv_detection/
├── images/          # Input images
├── output/          # Processed images and results
├── detect_leaves.py # Main script for leaf detection
├── requirements.txt # Python dependencies
└── README.md        # Project documentation
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](../LICENSE) file for details.

## Acknowledgments

- OpenCV documentation and tutorials
- Community contributions to computer vision research
- Inspiration from nature and plant biology
- Plantesa training initiative