# P-OLVILICAE: Contrastive Autoencoder for Pediatric OLVILI Prediction

## 🚀 Project Overview

**P-OLVILICAE** (Pediatric One-Lung Ventilation-Induced Lung Injury Contrastive AutoEncoder) is a deep learning framework specifically engineered to predict lung injury in pediatric patients undergoing thoracic surgery. 

One-lung ventilation-induced lung injury (**OLVILI**) is a severe, life-threatening complication. Early identification is crucial for optimizing intraoperative strategies, yet it remains a significant challenge in clinical practice due to the complexities of pediatric physiological data.

---

## ⚠️ The Problem

The clinical and data-driven challenges in this domain include:

* **Clinical Risk:** OLVILI is life-threatening but difficult to detect early in children due to their unique physiological responses.
* **Data Scarcity & Quality:** Pediatric clinical datasets often suffer from a limited number of labeled samples and significant **class imbalance**.
* **Feature Complexity:** Intraoperative clinical features are high-dimensional, non-linear, and highly heterogeneous, making them difficult for standard models to process.
* **Model Limitations:** Conventional machine learning models (e.g., Logistic Regression, SVM, Random Forest) often fail to extract robust, discriminative representations from such specialized and "noisy" clinical data.

---

## 💡 Our Solution: P-OLVILICAE



This project introduces a **Contrastive Autoencoder-based framework** that bridges the gap between unsupervised representation learning and supervised classification. 

### Key Innovations:
* **Hybrid Representation Learning:** Integrates reconstruction-based learning with **Supervised Contrastive Learning**. This allows the model to exploit both labeled and unlabeled samples to learn a more robust latent space.
* **Discriminative Latent Space:** By pulling similar clinical cases together and pushing dissimilar ones apart in the feature space, the model achieves superior class separation compared to traditional autoencoders.
* **Joint Optimization:** The framework employs a multi-task loss function to balance reconstruction accuracy and predictive power:
    $$L_{total} = \lambda_1 L_{recon} + \lambda_2 L_{supCon} + \lambda_3 L_{BCE}$$
* **Clinical Superiority:** Validated on real-world pediatric OLV datasets, P-OLVILICAE consistently outperforms baseline models in **F1-score** and **AUC**, maintaining an optimal balance between sensitivity and specificity.

---

## 🛠 Model Architecture

The framework consists of three core modules:
1.  **Encoder:** Maps high-dimensional clinical features into a low-dimensional latent representation.
2.  **Decoder:** Ensures the latent features retain essential information by reconstructing the original input.
3.  **Contrastive & Classification Head:** Refines the latent space using supervised contrastive loss and performs final binary classification for lung injury risk.
