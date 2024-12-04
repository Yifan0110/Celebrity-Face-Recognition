# Celebrity-Face-Recognition
This repository implements a celebrity face classification system using transfer learning with the DenseNet121 architecture. The model is fine-tuned to achieve high accuracy and generalization while addressing overfitting challenges, making it robust for multi-class classification tasks.

## Data Processing  

### **1. Face Extraction**  
- Used **MTCNN (Multi-Task Cascaded Convolutional Neural Networks)** to detect and extract faces from images.  

### **2. Data Augmentation**  
To improve the model's robustness and reduce overfitting, the following augmentation techniques were applied:  
- **Rotation**  
- **Shifting**  
- **Shear**  
- **Zoom**  
- **Flip**  
- **Fill Mode**  

### **3. Balancing Classes**  
- Ensured that each class had an equal number of images (400 images per class) to prevent class imbalance issues.  

### **4. Dataset Split**  
- **Train Dataset**: 80% of the total images.  
- **Validation Dataset**: 20% of the total images.

## Model Overview  

### **1. Initial Implementation**
To improve upon challenges observed in a ResNet50-based model, the DenseNet121 architecture was selected for its efficient feature propagation and feature reuse. The model was customized with the following additional layers:  
- **GlobalAveragePooling2D**: Condenses feature maps into a single vector per feature map, reducing spatial dimensions while retaining critical information.  
- **Dense Layer (256 Units)**: With ReLU activation to capture high-level patterns.  
- **Dropout Layer (Rate: 0.5)**: Prevents overfitting by randomly deactivating 50% of neurons during training.  
- **Output Dense Layer**: Number of units set to the number of classes (as determined from the training data) with softmax activation for multi-class classification.  

### **Initial Performance Metrics**
- **Train Accuracy**: 95.58%  
- **Validation Accuracy**: 94.41%  
- **F1-Score**: 0.7204  

---

## Fine-Tuning and Optimization  

To further enhance performance, the following techniques were applied:  

### **1. Unfreezing More Layers**
- Deeper layers of the pretrained DenseNet121 were unfrozen, allowing the model to fine-tune deeper feature representations specific to the dataset.  

### **2. Adding Fully Connected Layers and Regularization**
- Additional dense layers were introduced to capture intricate patterns.  
- Regularization techniques were applied to improve generalization and reduce overfitting.  

### **3. Learning Rate Optimization**
- **ReduceLROnPlateau Callback**: Dynamically adjusts the learning rate based on validation loss for efficient training.  
- **EarlyStopping Callback**: Halts training when no improvement in validation loss is observed to prevent overfitting.  

---

### **Performance After Fine-Tuning**  

| Metric            | Value     |  
|--------------------|-----------|  
| **Train Accuracy** | 98.99%    |  
| **Validation Accuracy** | 97.06% |  
| **F1-Score**       | 0.8640    |  

These improvements resulted in exceptional generalization and balanced precision and recall, making the fine-tuned DenseNet121 model a robust choice for celebrity recognition tasks.


