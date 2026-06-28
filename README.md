# 🩺 Deep Learning Framework for Clinical Cirrhosis Risk Diagnosis

## 📌 Business Case & Project Overview
Clinical decisions regarding chronic liver diseases require objective risk screening tools. This project utilizes the **Discovery-to-Action (DTA)** framework to process historical clinical health data and train a scale-invariant **Feedforward Artificial Neural Network (ANN)** via TensorFlow/Keras. The goal is to accurately distinguish between high-risk mortality profiles and stable survival vectors, providing an objective, secondary screening tool for healthcare practitioners.

---

## 🔍 Discovery Phase: Clinical Preprocessing & Feature Architecture

### 1. Missing Value and Record Isolation
To ensure absolute data integrity for backward error propagation inside the neural graph, all observations containing any incomplete fields or `NA` strings were removed. 

### 2. Algorithmic Leakage Isolation
The columns `ID`, `N_Days`, `Status`, and `Drug` were removed prior to model initialization. 
* Removing `ID` and `Drug` filters out observational noise and biases related to clinical trial assignments.
* Removing `N_Days` (the number of days between registration and data collection) prevents **feature leakage**, as temporal duration directly correlates with survival status but is unavailable during initial patient screening.

### 3. Structural Encoding Map Transformations
Categorical observations were mapped into clean numeric vectors to match the activation functions of the network:
* Target Conversion (`Status`): `D` (Death) $\rightarrow 0$; `C` and `CL` (Survival indicators) $\rightarrow 1$.
* Binary Character Mapping: `F` (Female) $\rightarrow 1$, `M` (Male) $\rightarrow 0$, `Y` (Yes) $\rightarrow 1$, `N` (No) $\rightarrow 0$.
* Multi-Class Categories: Converted via `pd.get_dummies()` to generate orthogonal binary flags for remaining indicators (e.g., `Edema`).

### 4. Euclidean Regularization via Standard Scaling
Because neural network weights optimize via gradient descent, variations in absolute feature scales distort backpropagation trajectories. The feature matrix was normalized using `StandardScaler`:

$$z = \frac{x - \mu}{\sigma}$$

This maps input values to a shared space with a mean of `0.0` and a standard deviation of `1.0`, preventing high-magnitude features (e.g., `Cholesterol` or `Alk_Phos`) from destabilizing gradient optimizations.

---

## 🛠️ Technical Phase: Deep Learning Architecture & Loss Dynamics

The diagnostic network was built using a structured sequential topology:
1. **Input Interface:** Dynamically maps features matching the encoded training data array dimensions.
2. **Hidden Layer 1:** 16 hidden neurons leveraging Rectified Linear Unit (`relu`) activations to model non-linear transformations.
3. **Hidden Layer 2:** 16 hidden neurons leveraging `relu` activations to isolate high-level feature combinations.
4. **Output Interface:** A single specialized hidden neuron utilizing a `sigmoid` activation function to output a calibrated risk probability map constrained between $[0, 1]$.

### Loss Minimization Objective
The structural weights were optimized over exactly **10 epochs** using the Adam gradient optimizer to minimize Binary Cross-Entropy Loss:

$$\text{Loss} = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

---

## 📊 Evaluation & Generalization Performance Matrix

The neural network achieved stable optimization convergence across its brief 10-epoch training cycle:

| Operational Metric | Final Value Achievement | Strategic Meaning & Clinical Scope |
|--------------------|-------------------------|------------------------------------|
| **Training Accuracy** | *~78.45%* | Initial parameters fit onto base stratified sample observations. |
| **Test Accuracy** | *~76.19%* | Verified generalization capacity across completely unseen patient profiles. |

👉 **Visual Artifact Reference:** Model convergence trends over the training lifecycle are saved as `nn_convergence_loss.png`.

---

## 🏢 Action Phase: Clinical Decision Risk & Error Analysis

### 🚨 Critical Trade-Off: False Positives vs. False Negatives
Before deploying any deep learning model into clinical practice, we must evaluate the ethical and operational impact of classification errors:
* **False Positives (Type I Error):** The model predicts high mortality risk for a stable patient. While this introduces psychological stress and triggers unnecessary diagnostic pipelines or invasive biopsies, the clinical outcome is bounded by follow-up human reviews.
* **False Negatives (Type II Error):** The model predicts stable survival for a patient at high risk of mortality. **This is the most critical failure mode in medicine.** A False Negative leads to a missed diagnosis, withholding life-saving clinical interventions, and allowing asymptomatic cirrhosis to progress unchecked.

👉 **Visual Artifact Reference:** Check `clinical_confusion_matrix.png` in the repository root directory to see the complete True vs. False classification distributions.

---

## 🏥 Medical Advisory Board Executive Recommendations

### 🚫 Clinical Readiness Proclamation
**The current network configuration is NOT approved for immediate, autonomous medical diagnostic deployment.** While a test accuracy of ~76% demonstrates strong pattern extraction capabilities, a ~24% error rate presents unacceptable risks to patient safety regarding missed diagnoses (False Negatives). This model should be treated strictly as an exploratory baseline secondary screening indicator.

### 🚀 Roadmap to Production-Grade Clinical Deployment
To upgrade this neural pipeline for safe, real-world hospital deployment, the following data engineering tasks must be executed next:
1. **Asymmetric Cost Class Weighting:** Modify the training loss function to penalize False Negatives 5x more severely than False Positives, steering the network's optimization path toward absolute sensitivity.
2. **Classification Threshold Optimization:** Calibrate the default decision threshold downward from $0.50$ using a Receiver Operating Characteristic (ROC) curve to deliberately maximize patient recall.
3. **External Cohort Validation:** Validate the model's performance on external patient data from separate medical facilities to ensure the network generalizes well across different demographics and equipment profiles.
