# 🩺 Pneumonia Detection using Deep Learning  

## 📌 Overview  
This project detects Pneumonia in chest X-ray images using Deep Learning. We use VGG19 architecture with transfer learning (pre-trained on ImageNet). Some layers are unfrozen for fine-tuning to improve model accuracy. The model is deployed using Flask and can predict X-ray images via an API.


## 🏗️ Model Architecture  
- **Base Model:** VGG19 (Pretrained on ImageNet)  
- **Transfer Learning:** Used pretrained weights, unfreezing initial layers  
- **Feature Extraction:** Extracted deep features from X-rays  
- **Framework:** TensorFlow/Keras  

## 🔧 Installation  
Clone the repository and install dependencies:  
```bash
git clone <your-repo-link>
cd pneumonia-detection
pip install -r requirements.txt

