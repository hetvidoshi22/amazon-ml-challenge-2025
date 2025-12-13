# ML Challenge 2025: Smart Product Pricing Solution Template

**Team Name:** Accuracy Alliance 
**Team Members:** Manav Shah, Hetvi Doshi, Pratham Tailor, Krishna Champaneria		
**Submission Date:** 13 October 2025

---

## 1. Executive Summary
*Our solution leverages a hybrid modeling approach that combines ResNet50V2 image embeddings with TF-IDF text features and engineered numerical attributes, trained using a regularized LightGBM regressor. To avoid overfitting—identified as the major challenge during experimentation—we implement a two-stage training strategy with early stopping, first finding the optimal number of boosting rounds on an 80/20 split and then retraining on 100% of the data using that exact iteration count. This ensures maximum generalization performance while retaining model simplicity, efficiency, and reproducibility.*
---

## 2. Methodology Overview

### 2.1 Problem Analysis
*The goal of this challenge was to predict product prices using both image data and catalog text. During our initial analysis, we observed that prices varied widely across different product types, and many entries contained incomplete or noisy information. We also noticed that advanced models tended to overfit, which confirmed that controlling model complexity would be more effective than simply increasing model size.*

**Key Observations:** *The price distribution was heavily skewed, with a few very high-priced products affecting overall model stability.The catalog text often contained structured patterns such as "Value" and "Unit", which could be extracted and used as reliable features.Images alone were not enough to determine prices and worked best when combined with text features.Many products had missing or unclear descriptions, so default placeholders were needed during preprocessing.Overfitting was the main challenge, as more complex models performed worse on unseen data.*

### 2.2 Solution Strategy
*We implemented a multimodal learning strategy that integrates image embeddings from ResNet50V2, textual features extracted via TF-IDF, and structured numerical/categorical attributes from catalog data. Instead of using ensembles or heavy hyperparameter tuning—which previously caused overfitting—we trained a single, regularized LightGBM model. To maximize generalization, we used a two-stage early stopping strategy: first determining the optimal number of boosting rounds on a validation split, then retraining on the full dataset with exactly that number of iterations. This approach ensures effective learning from all data modalities while minimizing overfitting and maintaining model simplicity.*

**Approach Type:** Single-Model Hybrid Learning (Multimodal Regression)
**Core Innovation:** nstead of relying on complex ensembles or aggressive tuning, we focused on controlled generalization by integrating image embeddings, text features, and structured signals within a regularized LightGBM regressor — trained using a two-stage early stopping strategy that identifies the ideal number of boosting rounds before retraining on full data. This allowed us to leverage multimodal richness without risking overfitting.

---

## 3. Model Architecture

### 3.1 Architecture Overview
*Our solution follows a multimodal regression pipeline, combining image, text, and structured features into a single LightGBM model. The architecture ensures that each modality contributes effectively while avoiding overfitting through a controlled training strategy.*

*Architecture Flow:*
           ┌───────────────┐
           │   Raw Data    │
           │ (Images +     │
           │  Catalog CSV) │
           └───────┬───────┘
                   │
    ┌──────────────┴──────────────┐
    │                             │
┌───────────────┐           ┌───────────────┐
│ Image Features│           │ Catalog Text  │
│ (ResNet50V2)  │           │ (TF-IDF)     │
└───────┬───────┘           └───────┬───────┘
        │                           │
        │                     ┌───────────────┐
        │                     │ Structured    │
        │                     │ Features      │
        │                     │ (Value, Unit) │
        │                     └───────┬───────┘
        │                             │
        └─────────────┬───────────────┘
                      │
            ┌─────────────────────┐
            │ Feature Concatenation│
            │ (Sparse + Dense)    │
            └─────────┬───────────┘
                      │
            ┌─────────────────────┐
            │ LightGBM Regressor  │
            │ (Regularized, Two-  │
            │ Stage Early Stopping)│
            └─────────┬───────────┘
                      │
            ┌─────────────────────┐
            │ Predicted Price     │
            └─────────────────────┘

### 3.2 Model Components

**Text Processing Pipeline:**
*The text processing pipeline begins by extracting structured fields from the catalog text, specifically the Value and Unit information. These fields are removed from the raw text to isolate the descriptive content, and any missing entries are replaced with the placeholder "missing". The text is then cleaned by converting to lowercase, removing stopwords, and trimming unnecessary whitespace. For modeling, we use a TF-IDF vectorizer that transforms the processed text into a sparse representation, with a maximum of 20,000 features, a minimum document frequency of 5, and bi-gram tokenization to capture short phrases.*

*The image processing pipeline leverages pre-trained ResNet50V2 for feature extraction. Images are first resized to match the model’s input dimensions and normalized to the [0,1] range. ResNet50V2 is used without its top classification layer, and global average pooling is applied to produce fixed-length embeddings of 2048 dimensions, capturing rich visual features for each product.*

*In addition to text and image data, structured features such as the numerical Value and categorical Unit are incorporated. Missing numerical values are replaced with the median, while missing categorical entries are labeled as "unknown". The numerical data is standardized using a scaler, and the categorical data is converted via one-hot encoding with an option to ignore unknown categories. All three feature types — text, image, and structured — are then combined into a single feature matrix, which serves as input to the LightGBM regressor.*
---
## 4. Model Performance

### 4.1 Validation Results
- **SMAPE Score:** 52.7894%

*The model was evaluated on an 80/20 train-validation split to estimate its generalization performance. Using the two-stage early stopping strategy, the LightGBM regressor achieved a SMAPE of 52.79% on the validation set. This indicates that the model captured general pricing patterns reasonably well while controlling for overfitting. Additional evaluation using metrics like Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) further confirmed that the predictions were consistent and stable across the diverse range of product prices, despite the presence of outliers and skewed distributions.*

## 5. Conclusion
*Our approach combined image embeddings from ResNet50V2, TF-IDF text features, and structured numerical attributes within a single, well-regularized LightGBM model. By implementing a two-stage early stopping strategy, we effectively controlled overfitting and achieved robust predictive performance, with a validation SMAPE of 52.79%. This experience reinforced the importance of disciplined model training and careful feature engineering when working with multimodal and noisy data.*
---
