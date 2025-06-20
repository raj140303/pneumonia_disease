# 🩺 Pneumonia Detection Using Deep Learning

> A Deep Learning based web application to detect Pneumonia from chest X-ray images. Just upload the image — and get the result: **Normal** or **Pneumonia** within seconds!

---

## 📌 Project Description

This project is a real-world implementation of a **Convolutional Neural Network (CNN)** to classify chest X-ray images. It is built using **Keras**, **TensorFlow**, and deployed with **Flask** as a web application.

The model has been trained on a publicly available dataset consisting of chest X-ray images classified as **NORMAL** or **PNEUMONIA** (which includes both bacterial and viral types but treated as one class in this binary classification task).

---

## 🚀 Key Features

✅ Upload chest X-ray images via a web interface  
✅ Predicts whether the person is **Normal** or has **Pneumonia**  
✅ Trained on a real-world dataset of over 5,000+ X-ray images  
✅ Built using TensorFlow and Flask for full stack deployment  
✅ Clean, user-friendly UI  

---

## 🧠 Model Architecture

- ✅ Input Size: `(150, 150, 3)`
- ✅ CNN Layers: Convolution → ReLU → MaxPooling
- ✅ Flatten + Dropout
- ✅ Output Layer: 1 neuron with **Sigmoid Activation**
- ✅ Binary Classification (0 = Normal, 1 = Pneumonia)

> The model uses **Binary Crossentropy** as the loss function and **Adam Optimizer** for training.

---

## 📊 Dataset Information

- **Dataset Name**: Chest X-Ray Images (Pneumonia)
- **Source**: [Kaggle - Paul Mooney's Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Total Images**: 5,867
  - NORMAL: 1,583
  - PNEUMONIA: 4,284 (bacterial and viral combined)

📌 **Note**: Although the raw dataset has separate folders for `bacteria` and `virus`, this project performs **binary classification**, not multi-class.

---
## 🖼️ Application Screenshots

### 🏠 Homepage (Before Uploading X-ray)
![Homepage](static/images/Home_page.png)

### ✅ Prediction: Normal Detected
![Normal Detected](static/images/normal_detected.png)

### ❌ Prediction: Pneumonia Detected
![Pneumonia Detected](static/images/pneumonia_detected.png)

---

## ⚙️ Tech Stack

| Layer         | Technology              |
|---------------|--------------------------|
| Frontend      | HTML, CSS (via Flask Templates) |
| Backend       | Python (Flask)           |
| Deep Learning | TensorFlow / Keras       |
| Data Handling | OpenCV, NumPy, Matplotlib |

---

## 🛠️ How to Run This Project Locally

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/pneumonia-detection.git
cd pneumonia-detection
