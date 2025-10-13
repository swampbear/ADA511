# Spam Detection OPM

## Overview
This project demonstrates using an Optimal Predictor Machine for text classification. The OPM learns word frequency distributions and uses Bayesian inference to classify emails as spam or legitimate (ham). This is conceptually similar to Naive Bayes classifiers.

## Key Concepts
- **Bag of Words**: Representing text as word frequency features
- **Conditional Independence**: Assuming words occur independently given the class
- **Laplace Smoothing**: Handling unseen words in test data
- **Likelihood Ratios**: Identifying words most indicative of spam vs ham

## Use Case
Given an email's text content, predict whether it is spam or legitimate based on the words it contains.

## How to Run
```bash
python3 spam_detection_opm.py
```

No external dependencies beyond NumPy, Pandas, and Matplotlib.

## What It Does
1. Generates synthetic email data with realistic spam/ham characteristics
2. Builds a vocabulary of the most common words
3. Trains an OPM to learn P(Word | Class) distributions
4. Classifies test emails using Bayesian inference
5. Identifies most informative words for spam detection
6. Creates comprehensive visualizations

## Output
- **Performance metrics**: Accuracy, precision, recall, F1-score
- **Informative words**: Which words are strongest spam/ham indicators
- **Confusion matrix**: Breakdown of correct and incorrect predictions
- **Visualizations**: Saved as `spam_detection_results.png`

## Learning Points
- How OPM handles text data through probabilistic modeling
- The connection between OPM and Naive Bayes
- Feature engineering for text (bag of words)
- Dealing with high-dimensional sparse features
- Smoothing techniques for probability estimation
