# 🚗 **Traffic Accident Cause Classification Using LightGBM**

### **Classifying the Etiology of NYC Motor Vehicle Collisions (2022–2024) Using Spatiotemporal Metadata + XAI**

This project builds an interpretable machine learning pipeline to classify the **primary contributing factors (“causes”) of vehicle collisions** using only environmental and spatiotemporal metadata.

The model moves beyond raw crash counts toward **causal classification**, enabling targeted public safety interventions, intersection redesign, and data-driven urban planning.

---

## 📌 **Dataset**

**Source:**
U.S. Government Open Data Portal
**Motor Vehicle Collisions – Crashes**
[https://catalog.data.gov/dataset/motor-vehicle-collisions-crashes](https://catalog.data.gov/dataset/motor-vehicle-collisions-crashes)

**Scope Used in This Project:**
Filtered to **2022–2024** to ensure modern and relevant trends.

### **Key Properties**

* **Size:** 2.2M+ total records; ~350k used after filtering to 2022–2024.
* **Severe class imbalance:**

  * *Driver Inattention/Distraction:* 57,729 instances
  * *Turning Improperly:* 5,448 instances
* **Top-10 contributing factors** retained; ambiguous categories like *“Unspecified”* removed.
* **Temporal patterns:**

  * Clear weekday peaks
  * Morning and evening commute spikes
* **Spatial patterns:**

  * Brooklyn and Queens consistently lead in crash volumes

### **Train/Test Split**

* **80% training**, **20% validation** (stratified on contributing factor)

---

## 🔍 **Problem Statement**

Motor vehicle collision data is large, noisy, and inconsistent. Simply counting crashes offers limited insight into *why* accidents occur.
This project focuses on **classifying the causal factor** behind a crash (e.g., *Unsafe Speed*, *Turning Improperly*) by analyzing:

* Time of crash
* Location / street name
* Borough
* Vehicle type
* Environmental metadata

The dataset suffers from **extreme class imbalance**, making the prediction of rare but important causes challenging.

---

## 💡 **Why This Matters**

Understanding *why* collisions occur is critical for:

### **1. Targeted Infrastructure Changes**

Different causes require different interventions:

* Improper turns → intersection redesign
* Unsafe speed → signal timing or speed cameras
* Lane usage issues → lane separators

### **2. Behavioral Enforcement**

Distinguishing engineering problems from driver cognition issues.

### **3. Filling Missing Labels**

Crash records frequently lack complete cause information.
A predictive model helps infer likely causes from metadata.

### **4. Policy Transparency**

Using XAI (SHAP + LIME), every prediction can be explained to planners and policymakers.

---

## 🧠 **Methods & Pipeline**

### **1. Resampling Strategy (Hybrid)**

To combat class imbalance:

* **Undersampling** the majority class to ~14,158 instances
* **Oversampling** minority classes to ~1.2× the mean (~16,989 instances)

This dramatically improved model sensitivity to rare labels.

---

### **2. Model Architecture: LightGBM**

**Hyperparameters**

* `objective='multiclass'` (10 classes)
* `learning_rate=0.05`
* `num_leaves=80`
* `min_data_in_leaf=60`
* `max_bin=255` (optimized for GPU)

**Training Setup**

* Early stopping (100 rounds patience)
* Monitored:

  * **Weighted F1-score** (primary metric)
  * `multi_logloss`

---

### **3. Explainability (XAI)**

The model integrates:

#### 🔹 **SHAP (Global Interpretability)**

Used to understand:

* Which features matter most
* How they push predictions toward specific causes

**Top Predictors:**
`CRASH_DATE`, `CRASH_TIME`, `LONGITUDE`, `ON STREET NAME`

#### 🔹 **LIME (Local Interpretability)**

Explains **individual crash predictions**, useful for case-by-case analysis.

---

## 📊 **Experimental Results**

### **Best Model Iteration:**

**2,707**

### **Overall Metrics**

| Metric                | Result    |
| --------------------- | --------- |
| **Weighted F1 Score** | **0.733** |
| **Multi-Logloss**     | ~0.87     |

### **Per-Class Accuracy**

| Contributing Factor Class          | Accuracy  | Tier     |
| ---------------------------------- | --------- | -------- |
| Backing Unsafely                   | 97.8%     | High     |
| Turning Improperly                 | 94.1%     | High     |
| Traffic Control Disregarded        | 91.3%     | High     |
| Other Vehicular                    | 89.1%     | High     |
| Passing Too Closely                | 85.7%     | High     |
| Unsafe Speed                       | 82.3%     | High     |
| Passing or Lane Usage Improper     | 71.2%     | Moderate |
| Following Too Closely              | 55.7%     | Moderate |
| Failure to Yield                   | 52.1%     | Moderate |
| **Driver Inattention/Distraction** | **18.9%** | **Low**  |

---

## 🧭 **Interpretation & Insights**

### **1. Objective Actions vs Subjective States**

The model excels at **physical maneuvers**:

* Improper turns
* Backing unsafely
* Traffic control violations

These have strong temporal/spatial signatures → easier to classify.

But it struggles with **cognitive states**:

* Driver inattention
* Failure to yield
* Following too closely

These categories are noisy, ambiguous, and often inconsistently labeled.

---

### **2. Limitations**

* Metadata alone cannot capture **driver cognition**
* “Driver Inattention” is a noise-heavy, catch-all label
* Many crashes have ambiguous, missing, or duplicate contributing factors
* Street names create high-cardinality challenges (partially mitigated)

---

## 🏙️ **Implications for Policy & Urban Planning**

This model is far more effective for **environmental risk detection** than for predicting human cognitive states.

Ideal applications:

* Identifying risky intersections
* Detecting streets with high improper-turn density
* Localized signal timing adjustments
* Planning lane separators and road geometry changes
* Prioritizing enforcement at identified hotspots

Not ideal for:

* Determining driver blame
* Modeling distractions or psychological attributions without richer data

---

## 🚀 **Future Work**

To improve classification of cognitive causes:

* Add **NLP embeddings** from police narratives
* Incorporate **vehicle telematics** or traffic camera metadata
* Use **Graph Neural Networks** for road network topology
* Multi-label classification (many crashes have multiple causes)

---

## 🔗 **Repository**

GitHub: **[https://github.com/utki007/traffic-accident-ml-classifier](https://github.com/utki007/traffic-accident-ml-classifier)**

---

## 📜 **License**

MIT License—free for academic and industry use.

---
