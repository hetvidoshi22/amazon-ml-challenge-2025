# Amazon ML Challenge 2025 – Multimodal Price Prediction

This repository contains our solution for the **Amazon ML Challenge 2025**, where the task was to predict product prices using **multimodal data** (images, text, and structured attributes).

---

## 🏆 Overview
- **Team:** Accuracy Alliance  
- **Final Rank:** 1470 / 6000+ teams  
- **Metric:** SMAPE  
- **Approach:** Multimodal Regression

---

## 🧠 Approach
- Built a **multimodal ML pipeline** combining:
  - Image embeddings from **ResNet50V2**
  - Text features using **TF-IDF**
  - Structured numerical and categorical attributes
- Trained a **regularized LightGBM regressor**
- Used **early stopping** to control overfitting and improve generalization

---

## 📊 Results
- **Validation SMAPE:** ~52.78%
- Stable performance across diverse and noisy product catalogs

---

## 🛠️ Tech Stack
- **Language:** Python  
- **Models:** LightGBM, ResNet50V2  
- **Text Processing:** TF-IDF  
- **Libraries:** scikit-learn, LightGBM 
- **Platform:** Google Colab, VS Code
