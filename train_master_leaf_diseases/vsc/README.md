# Leaf Disease Detection Training (Visual Studio Code)

This repository contains a project for training a model to detect leaf diseases using machine learning and computer vision. The training process is conducted using Visual Studio Code.

## Features
- Image preprocessing and augmentation.
- Model training for disease classification.
- Visual Studio Code-based workflow for flexibility and efficiency.

## Prerequisites
- Python 3.x
- Visual Studio Code
- Required Python libraries (see `requirements.txt`)

## Setup Instructions
1. Clone the repository:
    ```bash
    git clone https://github.com/theonegareth/Plantesa-training.git
    cd Plantesa-training/leaf_disease_detection
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Open the project in Visual Studio Code:
    ```bash
    code .
    ```

4. Configure the Python environment:
    - Ensure the Python extension is installed in Visual Studio Code.
    - Select the appropriate Python interpreter for the project.

5. Run the training script:
    - Navigate to the `src/` directory and execute the training script using the integrated terminal.

## Project Structure
- `data/`: Contains datasets for training and testing.
- `models/`: Pre-trained and custom models for disease detection.
- `src/`: Source code for preprocessing and training.
- `README.md`: Project documentation.

## Flowchart

Below is the flowchart illustrating the training process:

![Teacher Training Flowchart](assets/Teacher_Training_Flowchart.png)

1. **Start**: The process begins with the initialization of the training pipeline.

2. **Data Preparation**: The raw data is collected and preprocessed to ensure it is suitable for training. This step may include cleaning, normalization, and feature extraction.

3. **Data Splitting**: The prepared data is split into training, validation, and testing datasets to evaluate the model's performance effectively.

4. **Data Augmentation**: Techniques such as rotation, flipping, and scaling are applied to artificially increase the size and diversity of the training dataset.

5. **Model Architecture**: The machine learning model is defined, including its layers, activation functions, and other hyperparameters.

6. **Train for n Epochs**: The model is trained for a specified number of epochs using the training dataset.

7. **Evaluate Accuracy**: After each epoch, the model's accuracy is evaluated using the validation dataset. If the accuracy improves, the training continues; otherwise, the process moves to the next step.

8. **Generate Data for Evaluation**: Once training is complete, the model generates predictions on the testing dataset for evaluation purposes.

9. **Accuracy and Loss Chart**: A graph is created to visualize the model's accuracy and loss over the training epochs.

10. **Confusion Matrix**: A confusion matrix is generated to analyze the model's classification performance.

11. **Normalized Confusion Matrix**: A normalized confusion matrix is created to provide a clearer view of the model's performance across different classes.

12. **Save Data/Graph to Another Notebook**: All generated data and graphs are saved to another notebook for further analysis and documentation.

13. **Stop**: The process ends, and the trained model is ready for deployment or further evaluation.


## Usage
1. Place leaf images in the `data/input` directory.
2. Open the training script in Visual Studio Code and follow the steps to train the model.
3. Save the trained model for future use.

## Acknowledgments
- Open-source libraries and datasets used in this project.
- The Visual Studio Code community for their support.
