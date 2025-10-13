# Sentiment Analysis OPM

## Overview
This project demonstrates using an Optimal Predictor Machine for multi-class sentiment classification of product reviews. Unlike binary classification, this handles ordinal outcomes (1-5 stars) and shows how to compute expected values from probability distributions.

## Key Concepts
- **Multi-class Classification**: Predicting one of multiple ordered categories
- **Ordinal Outcomes**: Sentiment levels have natural ordering (1 < 2 < 3 < 4 < 5)
- **Expected Values**: Computing continuous estimates from discrete probabilities
- **Characteristic Words**: Identifying which words signal different sentiments
- **Uncertainty Quantification**: Using probability distributions to express confidence

## Use Case
An e-commerce platform wants to:
1. Automatically classify product reviews by sentiment (1-5 stars)
2. Understand which words indicate positive vs negative experiences
3. Compute expected ratings rather than just hard classifications
4. Identify uncertain predictions that may need human review

## How to Run
```bash
python3 sentiment_analysis_opm.py
```

## What It Does
1. Generates synthetic product reviews with realistic sentiment patterns
2. Trains an OPM to learn word-sentiment associations
3. Predicts sentiment probabilities for new reviews
4. Computes expected ratings as weighted averages
5. Identifies characteristic words for each sentiment level
6. Creates comprehensive visualizations showing:
   - Distribution of sentiments (true vs predicted)
   - Prediction confidence levels
   - Most characteristic words for each sentiment

## Output
- **Accuracy metrics**: Exact match and within-1-star accuracy
- **Characteristic words**: Words most indicative of each sentiment
- **Expected ratings**: Continuous estimates (e.g., 3.7 stars) vs discrete (4 stars)
- **Probability distributions**: Full distribution over all sentiment levels
- **Visualizations**: Saved as `sentiment_analysis_results.png`

## Learning Points
- How OPM handles ordinal multi-class problems
- The value of probability distributions beyond hard predictions
- Computing expected values for ordinal outcomes
- Feature importance in text classification
- When discrete predictions vs continuous estimates are more appropriate
- Quantifying prediction uncertainty

## Extension Ideas
- Add aspect-based sentiment (sentiment about specific product features)
- Include negation handling ("not good" vs "good")
- Model sentiment intensity beyond 5 levels
- Implement active learning (request labels for uncertain predictions)
- Add contextual features (product category, price range)
- Compare expected ratings vs mode predictions for decision making
