# Amazon ML Challenge 2025 – Multimodal Price Prediction

This repository contains our solution for the **Amazon ML Challenge 2025**, where the objective was to predict product prices using **multimodal data**, including images, catalog text, and structured attributes.

---

## 🏆 Overview
- **Team:** Accuracy Alliance  
- **Final Rank:** 1470 / 6000+ teams  
- **Evaluation Metric:** SMAPE  
- **Approach:** Multimodal Regression  

---

## 🧠 Approach
- Built a **multimodal ML pipeline** for product price prediction combining:
  - Image embeddings extracted using **ResNet50V2**
  - Text features generated via **TF-IDF**
  - Structured numerical and categorical attributes from catalog data
- Trained a **regularized LightGBM regressor** for price prediction
- Applied **early stopping** to control overfitting and improve generalization on unseen data

---

## 📊 Results
- **Validation SMAPE:** ~52.78%  
- Achieved stable performance across diverse and noisy product catalogs
- Final model ranked competitively on the public leaderboard

---

## 🛠️ Tech Stack
- **Language:** Python  
- **Models:** LightGBM, ResNet50V2  
- **Text Processing:** TF-IDF  
- **Libraries:** scikit-learn, LightGBM, TensorFlow
- **Platform:** Google Colab, VS Code  

---

## 📁 Repository Notes
- The dataset, extracted features, and trained model files are **not included** due to competition constraints and file size limitations.
- This repository focuses on **model architecture, feature engineering, and training logic**.

To reproduce the workflow:
1. Download the dataset from the official Amazon ML Challenge platform.
2. Place the files inside a local `dataset/` directory.
3. Follow the training and inference logic defined in the code.

---

## 👥 Team
**Accuracy Alliance**
- Manav Shah  
- Hetvi Doshi  
- Pratham Tailor  
- Krishna Champaneria  

---

## 📌 Key Takeaways
- Multimodal feature integration improves prediction only when model complexity is carefully controlled
- Regularization and early stopping were more effective than aggressive hyperparameter tuning
- Feature engineering played a crucial role in handling noisy, real-world catalog data
