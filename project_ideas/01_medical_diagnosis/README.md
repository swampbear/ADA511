# Medical Diagnosis OPM

## Overview
This project demonstrates using an Optimal Predictor Machine (OPM) for medical diagnosis. The OPM learns conditional probability distributions P(Disease | Symptoms) from patient data and uses Bayesian inference to predict diseases based on observed symptoms.

## Key Concepts
- **Bayesian Inference**: Uses Bayes' theorem to compute P(Disease | Symptoms)
- **Mutual Information**: Ranks symptoms by their diagnostic value
- **Conditional Probability**: Models P(Symptoms | Disease) from training data

## Use Case
Given a patient's symptoms (fever, cough, fatigue, etc.), predict which disease they most likely have (Flu, COVID-19, Common Cold, or Allergies).

## How to Run
```bash
python3 medical_diagnosis_opm.py
```

## What It Does
1. Generates synthetic patient data with realistic symptom-disease correlations
2. Trains an OPM to learn conditional probability distributions
3. Predicts diseases for new patients
4. Calculates mutual information to rank symptom importance
5. Creates visualizations showing model performance and insights

## Output
- **Accuracy metrics**: Shows prediction accuracy on test data
- **Symptom rankings**: Which symptoms are most informative for diagnosis
- **Prediction probabilities**: Full probability distribution for each prediction
- **Visualizations**: Saved as `diagnosis_results.png`

## Learning Points
- How OPM uses probability distributions for classification
- The role of mutual information in feature importance
- Bayesian inference in practice
- Difference between hard predictions and probability distributions
