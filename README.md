
# 🚦 Predicting Common Causes of Vehicle Collisions with Causal Analysis of Environmental Factors

[Dataset](https://catalog.data.gov/dataset/motor-vehicle-collisions-crashes)

## 🧠 Project Overview

Motor vehicle collisions are one of the most serious public safety issues worldwide, caused by behavioral, environmental, and infrastructural factors. Understanding why these crashes occur is vital for developing preventive strategies.

This project aims to:

1. Predict the most probable causes of crashes—such as distracted driving, speeding, or mechanical failure—based on contextual data like location, time, and vehicle type.  
2. Conduct causal analysis to evaluate how external factors like weather and lighting conditions influence crash frequency and severity.  
3. Provide interpretable insights using Explainable AI methods (LIME and SHAP) to ensure transparency and practical applicability.

Unlike prior studies that focus on crash frequency or severity, this project addresses the **multi-factor nature of crash causation**, treating it as a **multi-label classification problem** where each incident can have multiple contributing causes.

---

## 📊 Dataset

We use the **Motor Vehicle Collisions (Crashes)** dataset, publicly available from the U.S. Government Open Data portal:

[Motor Vehicle Collisions Dataset](https://catalog.data.gov/dataset/motor-vehicle-collisions-crashes)

- Millions of records spanning multiple years  
- Features include date, time, borough, coordinates, number of injuries/fatalities, vehicle type, and multiple contributing factor fields (e.g., driver inattention, unsafe speed, brake failure)  
- Records lighting and weather conditions for causal analysis  
- Training data: 2022–2024  

---
