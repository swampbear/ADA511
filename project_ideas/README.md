# Optimal Predictor Machine (OPM) Project Ideas

This directory contains inspiring project ideas demonstrating various applications of Optimal Predictor Machines. Each project is self-contained, well-documented, and easy to run.

## Overview

These projects showcase different aspects of OPM:
- **Inference**: Learning probability distributions from data
- **Decision-Making**: Choosing optimal actions based on predictions
- **Feature Analysis**: Understanding which features are most informative
- **Uncertainty Quantification**: Expressing confidence in predictions

All implementations are in Python with minimal dependencies (NumPy, Pandas, Matplotlib).

---

## Projects

### 1. Medical Diagnosis OPM
**Directory**: `01_medical_diagnosis/`
**Run**: `python3 01_medical_diagnosis/medical_diagnosis_opm.py`

**What It Does**: Predicts diseases based on patient symptoms using Bayesian inference.

**Key Features**:
- Binary symptom features (fever, cough, fatigue, etc.)
- Multiple disease classification (Flu, COVID-19, Cold, Allergies)
- Mutual information for symptom importance ranking
- Full probability distributions over diseases

**Learn About**:
- Basic OPM inference with discrete features
- Bayesian reasoning in medical diagnosis
- Feature importance via mutual information
- Multi-class classification with OPM

**Complexity**: ⭐ Beginner-friendly

---

### 2. Spam Detection OPM
**Directory**: `02_spam_detection/`
**Run**: `python3 02_spam_detection/spam_detection_opm.py`

**What It Does**: Classifies emails as spam or legitimate based on word frequencies.

**Key Features**:
- Text processing with bag-of-words representation
- Vocabulary building and feature extraction
- Laplace smoothing for unseen words
- Likelihood ratios for identifying spam indicators

**Learn About**:
- OPM with text data and high-dimensional features
- The connection between OPM and Naive Bayes
- Handling sparse features
- Text feature engineering

**Complexity**: ⭐⭐ Intermediate

---

### 3. Customer Churn Prediction OPM
**Directory**: `03_customer_churn/`
**Run**: `python3 03_customer_churn/customer_churn_opm.py`

**What It Does**: Predicts customer churn AND decides optimal retention strategies.

**Key Features**:
- Combines inference with decision-making
- Expected utility computation for actions
- Cost-benefit analysis of interventions
- Feature discretization for continuous variables

**Learn About**:
- Complete OPM system (inference + decision-making)
- Expected utility theory
- Action selection under uncertainty
- Business applications of probabilistic reasoning
- When prediction accuracy ≠ decision quality

**Complexity**: ⭐⭐⭐ Advanced

---

### 4. Sentiment Analysis OPM
**Directory**: `04_sentiment_analysis/`
**Run**: `python3 04_sentiment_analysis/sentiment_analysis_opm.py`

**What It Does**: Classifies product reviews into 5 sentiment levels (ordinal classification).

**Key Features**:
- Multi-class ordinal outcomes (1-5 stars)
- Expected rating computation from probabilities
- Characteristic words for each sentiment level
- Uncertainty quantification in predictions

**Learn About**:
- OPM with ordinal outcomes
- Computing expected values from probability distributions
- Multi-class text classification
- When to use continuous estimates vs discrete predictions
- Identifying uncertain predictions

**Complexity**: ⭐⭐ Intermediate

---

## Quick Start

Each project can be run independently:

```bash
# Medical Diagnosis
python3 01_medical_diagnosis/medical_diagnosis_opm.py

# Spam Detection
python3 02_spam_detection/spam_detection_opm.py

# Customer Churn
python3 03_customer_churn/customer_churn_opm.py

# Sentiment Analysis
python3 04_sentiment_analysis/sentiment_analysis_opm.py
```

Each script will:
1. Generate synthetic training data
2. Train an OPM model
3. Make predictions on test data
4. Display results and metrics
5. Save visualizations to the project directory

## Requirements

All projects use standard Python libraries:
- `numpy` - Numerical computations
- `pandas` - Data manipulation
- `matplotlib` - Visualizations

No additional dependencies required.

## Project Comparison

| Project | Problem Type | Features | Classes | Decision-Making | Difficulty |
|---------|-------------|----------|---------|----------------|------------|
| Medical Diagnosis | Multi-class | Binary (symptoms) | 4 diseases | No | ⭐ |
| Spam Detection | Binary | Text (word counts) | 2 (spam/ham) | No | ⭐⭐ |
| Customer Churn | Binary + Actions | Mixed (continuous + categorical) | 2 (churn/stay) | **Yes** | ⭐⭐⭐ |
| Sentiment Analysis | Multi-class ordinal | Text (word counts) | 5 (1-5 stars) | No | ⭐⭐ |

## Key Concepts Demonstrated

### Inference Techniques
- **Conditional Probability**: All projects learn P(Y | X) from data
- **Bayes' Theorem**: Converting P(X | Y) and P(Y) to P(Y | X)
- **Feature Independence**: Simplifying joint distributions
- **Smoothing**: Handling unseen feature values

### Analysis Methods
- **Mutual Information**: Measuring feature importance (Projects 1, 3)
- **Likelihood Ratios**: Identifying characteristic features (Projects 2, 4)
- **Expected Values**: Computing continuous estimates from probabilities (Project 4)
- **Feature Discretization**: Handling continuous variables (Project 3)

### Decision-Making (Project 3 Only)
- **Expected Utility**: Computing value of actions
- **Action Selection**: Choosing optimal strategies
- **Cost-Benefit Analysis**: Balancing intervention costs with outcomes

### Practical Considerations
- **Data Generation**: Creating realistic synthetic datasets
- **Train-Test Splits**: Proper evaluation methodology
- **Visualizations**: Communicating results effectively
- **Uncertainty**: Expressing confidence in predictions

## Choosing a Starting Project

**If you're new to OPM**: Start with **Medical Diagnosis** (Project 1)
- Simplest features (binary symptoms)
- Clear interpretation of results
- Easy to understand probability calculations

**If you're interested in text/NLP**: Try **Spam Detection** (Project 2)
- Learn text feature engineering
- See OPM applied to high-dimensional data
- Understand bag-of-words models

**If you want business applications**: Explore **Customer Churn** (Project 3)
- See complete inference + decision-making pipeline
- Learn expected utility computation
- Understand when predictions aren't enough

**If you like ordinal data**: Check out **Sentiment Analysis** (Project 4)
- Multi-class classification
- Expected value computation
- Uncertainty quantification

## Extension Ideas

Each project README includes specific extension ideas. General possibilities:

1. **Add Real Data**: Replace synthetic data with real datasets
2. **Feature Engineering**: Add more sophisticated features
3. **Model Comparison**: Compare OPM with other classifiers (Naive Bayes, Logistic Regression, Neural Networks)
4. **Active Learning**: Select most informative samples for labeling
5. **Calibration**: Ensure predicted probabilities match true frequencies
6. **Cross-Validation**: More robust evaluation
7. **Hyperparameter Tuning**: Optimize vocabulary size, smoothing, discretization bins
8. **Sequential Decision Making**: Multi-step decision problems
9. **Cost-Sensitive Learning**: Different costs for different types of errors
10. **Explainability**: Add feature attribution methods

## Understanding OPM

The Optimal Predictor Machine is based on:

1. **Probability Theory**: Everything is uncertain, model it with probabilities
2. **Bayesian Inference**: Learn from data using Bayes' theorem
3. **Decision Theory**: Make optimal decisions using expected utility
4. **Information Theory**: Measure feature importance with mutual information

**Key Advantages**:
- Provides full probability distributions (not just point predictions)
- Naturally handles uncertainty
- Easy to interpret (conditional probabilities)
- Can incorporate prior knowledge
- Combines seamlessly with decision-making

**When to Use OPM**:
- You need probability distributions (not just predictions)
- Interpretability is important
- You have limited training data
- You want to incorporate domain knowledge
- You need to make decisions under uncertainty

## Further Reading

For more on OPM theory and applications, see:
- Course materials in the main `ADA511` directory
- Lecture notes on inference and decision-making
- The OPM implementation in `code/OPM-nominal/`

## Questions or Issues?

Each project is designed to be self-explanatory with extensive comments and documentation.

Key files in each project:
- `<project_name>_opm.py` - Main implementation with detailed comments
- `README.md` - Project-specific documentation
- `*_results.png` - Generated visualizations (after running)

Happy learning!
