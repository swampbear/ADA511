# Customer Churn Prediction OPM

## Overview
This project demonstrates a complete OPM system that combines **inference** (predicting customer churn) with **decision-making** (choosing optimal retention strategies). It shows how to use probability predictions to maximize expected utility through business actions.

## Key Concepts
- **Inference**: Learning P(Churn | Features) from customer behavior
- **Decision Making**: Choosing actions to maximize expected utility
- **Expected Utility**: Computing value of actions considering costs and benefits
- **Feature Discretization**: Handling continuous features in OPM
- **Customer Lifetime Value**: Quantifying the value of retention

## Use Case
A telecommunications company wants to:
1. Predict which customers are likely to leave (churn)
2. Decide what retention strategy to use for each customer
3. Maximize profit by balancing intervention costs with retention value

## How to Run
```bash
python3 customer_churn_opm.py
```

## What It Does
1. Generates realistic customer data with behavioral and demographic features
2. Trains an OPM to predict churn probability
3. Ranks features by their predictive power
4. For each customer, computes expected utility of different retention actions:
   - Do nothing (no cost)
   - Offer discount ($20 cost, 40% effective)
   - Reach out personally ($15 cost, 30% effective)
   - Premium offer ($50 cost, 60% effective)
5. Recommends the action that maximizes expected utility
6. Creates comprehensive visualizations

## Output
- **Churn predictions**: Probability that each customer will leave
- **Feature importance**: Which customer attributes best predict churn
- **Optimal strategies**: Recommended action for each customer
- **Utility analysis**: Expected value of different retention strategies
- **Visualizations**: Saved as `churn_analysis_results.png`

## Learning Points
- How OPM combines inference and decision-making
- The difference between prediction accuracy and decision quality
- Computing expected utilities with uncertain outcomes
- Feature engineering: discretization of continuous variables
- Real-world business application: balancing costs and benefits
- Why different customers warrant different strategies

## Extension Ideas
- Add more customer features (demographics, usage patterns)
- Implement sequential decision making (multiple touchpoints)
- Include uncertain costs (e.g., discount acceptance rate)
- Add exploration vs exploitation (A/B testing strategies)
- Model customer segments with different lifetime values
