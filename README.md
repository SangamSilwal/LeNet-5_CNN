# LeNet-5 on MNIST (TensorFlow/Keras)

This project implements the **original LeNet-5 architecture** (Yann LeCun, 1998) using **TensorFlow/Keras** and trains it on the **MNIST handwritten digits dataset**. The model achieves approximately **98–99% accuracy**.

---

## Project Goal

To recreate the classical **LeNet-5 CNN architecture** and use it to classify handwritten digits (0–9) from the MNIST dataset.

---

**LeNet-5 Architecture (1998):**
- Input: 32×32 grayscale image  
- Activation: `tanh`  
- Pooling: `AveragePooling2D`  
- Output classes: 10  

### Layer Summary
- C1: Conv2D – 6 filters, 5×5  
- S2: Average Pooling  
- C3: Conv2D – 16 filters, 5×5  
- S4: Average Pooling  
- C5: Dense – 120 units  
- F6: Dense – 84 units  
- Output: Dense – 10 units, softmax  

---
![LeNet-5 Architecture](https://en.wikipedia.org/wiki/LeNet#/media/File:LeNet-5_architecture.svg)
