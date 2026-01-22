# Car Damage Detection using CNN

## Overview
This project implements a **Convolutional Neural Network (CNN)** to automatically detect and classify **car damages** from images. It is designed for **deep learning enthusiasts** or anyone working in **computer vision for automotive applications**.

Using this system, you can:
- Classify car damage types (e.g., scratch, dent, broken parts)
- Train your own CNN models with custom datasets
- Evaluate the performance of the model using standard metrics

---

## Dataset
- The dataset consists of **car images with annotated damage labels**.
- Images are organized into folders based on **damage type**.
- **Note:** Large datasets are **not included** in this repository. Please upload your dataset in a `data/` folder or use a cloud link.

---

## Methodology
1. **Data Preprocessing**
   - Images are resized and normalized.
   - Data augmentation is applied (rotation, flipping, scaling) to increase dataset diversity.

2. **CNN Model**
   - The model consists of multiple **Convolutional + Pooling layers**.
   - Fully connected layers at the end perform classification.
   - **Activation function:** ReLU for hidden layers, Softmax for output.

3. **Training**
   - Loss function: **Categorical Crossentropy**
   - Optimizer: **Adam**
   - Metrics: Accuracy, Precision, Recall

4. **Evaluation**
   - Model performance is evaluated on a **separate validation/test set**.
   - Confusion matrix and accuracy curves are plotted.
  
---

## Usage
1. Clone this repository:
   ```bash
   git clone https://github.com/JeevanaMohanKarani/Car-damage-detection-using-CNN.git

2.Install dependencies:
- pip install -r requirements.txt
  
3.Run the notebook car_damage_detection.ipynb to:
- Load dataset
- Train the CNN model
- Evaluate performance

---

## Results
- The model is capable of detecting car damage with high accuracy.
- Example predictions:
- Input: Car image → Output: Damage type label

---

## Dependencies
- Python 3.8+
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- OpenCV
- scikit-learn

---

## Future Improvements
- Increase dataset size for better accuracy
- Use transfer learning (e.g., VGG16, ResNet) for faster and better performance
- Deploy as a web app or mobile app for real-time damage detection

---

## Author

Jeevana Mohan Karani

GitHub: https://github.com/JeevanaMohanKarani
