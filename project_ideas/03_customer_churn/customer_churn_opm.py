"""
Customer Churn Prediction OPM - Optimal Predictor Machine for Retention

This example demonstrates using OPM for predicting customer churn based on
behavioral and demographic features. Includes decision-making with utilities
to optimize business interventions.

Use case: Predicting which customers will leave and deciding optimal retention strategies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class CustomerChurnOPM:
    """
    Optimal Predictor Machine for customer churn prediction with decision-making.

    Combines inference (predicting churn) with decision-making (choosing interventions)
    based on expected utilities.
    """

    def __init__(self, discretize_bins=5):
        """
        Parameters:
        -----------
        discretize_bins : int
            Number of bins for discretizing continuous features
        """
        self.discretize_bins = discretize_bins
        self.prior = {}  # P(Churn)
        self.conditional = {}  # P(Feature | Churn)
        self.feature_names = []
        self.feature_bins = {}  # Store bin edges for discretization

    def _discretize_feature(self, values, feature_name, fit=False):
        """
        Discretize continuous features into bins.

        Parameters:
        -----------
        values : array-like
            Feature values to discretize
        feature_name : str
            Name of the feature
        fit : bool
            If True, compute and store bin edges
        """
        if fit:
            # Compute quantile-based bins
            self.feature_bins[feature_name] = np.percentile(
                values, np.linspace(0, 100, self.discretize_bins + 1)
            )

        # Discretize using stored bins
        bins = self.feature_bins[feature_name]
        discretized = np.digitize(values, bins[1:-1])  # Returns values from 0 to discretize_bins-1

        return discretized

    def fit(self, X, y):
        """
        Learn conditional probability distributions from training data.

        Parameters:
        -----------
        X : DataFrame with customer features
        y : Series with churn labels (0 = stayed, 1 = churned)
        """
        self.feature_names = X.columns.tolist()

        # Calculate prior P(Churn)
        total = len(y)
        for churn_value in [0, 1]:
            self.prior[churn_value] = (y == churn_value).sum() / total

        # Discretize features and calculate P(Feature | Churn)
        X_discrete = pd.DataFrame()
        for feature in self.feature_names:
            X_discrete[feature] = self._discretize_feature(X[feature].values, feature, fit=True)

        # Calculate conditional probabilities
        self.conditional = {}
        for churn_value in [0, 1]:
            churn_mask = (y == churn_value)
            churn_data = X_discrete[churn_mask]

            self.conditional[churn_value] = {}

            for feature in self.feature_names:
                feature_counts = churn_data[feature].value_counts()
                total_samples = len(churn_data)

                # Laplace smoothing
                self.conditional[churn_value][feature] = {}
                for bin_idx in range(self.discretize_bins):
                    count = feature_counts.get(bin_idx, 0)
                    prob = (count + 1) / (total_samples + self.discretize_bins)
                    self.conditional[churn_value][feature][bin_idx] = prob

    def predict_proba(self, X):
        """
        Predict churn probability for each customer.

        Returns:
        --------
        DataFrame with P(Churn=0) and P(Churn=1)
        """
        predictions = []

        # Discretize features using stored bins
        X_discrete = pd.DataFrame()
        for feature in self.feature_names:
            X_discrete[feature] = self._discretize_feature(X[feature].values, feature, fit=False)

        for idx, customer in X_discrete.iterrows():
            probs = {}

            for churn_value in [0, 1]:
                # Start with prior
                prob = self.prior[churn_value]

                # Multiply by conditional probabilities (assuming independence)
                for feature in self.feature_names:
                    bin_idx = int(customer[feature])
                    prob *= self.conditional[churn_value][feature][bin_idx]

                probs[churn_value] = prob

            # Normalize
            total = sum(probs.values())
            if total > 0:
                probs = {k: v/total for k, v in probs.items()}

            predictions.append(probs)

        df = pd.DataFrame(predictions)
        df.columns = ['P(Stay)', 'P(Churn)']
        return df

    def predict(self, X):
        """Predict binary churn outcome (0=stay, 1=churn)."""
        proba = self.predict_proba(X)
        return (proba['P(Churn)'] > 0.5).astype(int)

    def mutual_information(self, feature):
        """Calculate mutual information between feature and churn."""
        mi = 0.0

        for churn_value in [0, 1]:
            p_churn = self.prior[churn_value]

            for bin_idx in range(self.discretize_bins):
                # P(Feature, Churn)
                p_joint = self.conditional[churn_value][feature][bin_idx] * p_churn

                if p_joint > 0:
                    # P(Feature)
                    p_feature = sum(
                        self.conditional[c][feature][bin_idx] * self.prior[c]
                        for c in [0, 1]
                    )

                    if p_feature > 0:
                        mi += p_joint * np.log2(p_joint / (p_feature * p_churn))

        return mi

    def rank_features(self):
        """Rank features by their predictive power (mutual information)."""
        mi_scores = {feature: self.mutual_information(feature) for feature in self.feature_names}
        return sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)


class ChurnDecisionMaker:
    """
    Decision maker that chooses optimal retention strategies based on utilities.

    Actions:
    - Do nothing
    - Offer discount
    - Reach out (personalized contact)
    - Premium offer
    """

    def __init__(self):
        # Cost of each action
        self.action_costs = {
            'do_nothing': 0,
            'offer_discount': 20,
            'reach_out': 15,
            'premium_offer': 50
        }

        # Probability of preventing churn given action (effectiveness)
        self.action_effectiveness = {
            'do_nothing': 0.0,
            'offer_discount': 0.4,
            'reach_out': 0.3,
            'premium_offer': 0.6
        }

        # Value of retaining a customer (customer lifetime value)
        self.retention_value = 200

    def compute_expected_utility(self, churn_prob, action):
        """
        Compute expected utility of an action for a customer.

        Utility = P(retain | action) * retention_value - action_cost
        """
        action_cost = self.action_costs[action]
        effectiveness = self.action_effectiveness[action]

        # Probability of retaining customer if we take this action
        if churn_prob < 0.5:
            # Already likely to stay
            p_retain = 1.0 - churn_prob
        else:
            # Likely to churn, action might prevent it
            p_retain = (1 - churn_prob) + churn_prob * effectiveness

        # Expected utility
        expected_value = p_retain * self.retention_value
        utility = expected_value - action_cost

        return utility

    def choose_best_action(self, churn_prob):
        """
        Choose the action that maximizes expected utility.

        Returns:
        --------
        action : str
            Best action to take
        utilities : dict
            Utility of each action
        """
        utilities = {}
        for action in self.action_costs.keys():
            utilities[action] = self.compute_expected_utility(churn_prob, action)

        best_action = max(utilities, key=utilities.get)
        return best_action, utilities


def generate_synthetic_customer_data(n_samples=1000):
    """
    Generate synthetic customer data with realistic churn patterns.
    """
    np.random.seed(42)

    data = []
    labels = []

    for _ in range(n_samples):
        # Generate customer features
        tenure_months = np.random.exponential(24)  # Time as customer
        monthly_charges = np.random.uniform(20, 150)
        total_charges = tenure_months * monthly_charges + np.random.normal(0, 100)
        support_calls = np.random.poisson(2)
        contract_type = np.random.choice([0, 1, 2])  # 0=month-to-month, 1=1-year, 2=2-year
        payment_method = np.random.choice([0, 1, 2, 3])  # Different payment methods
        internet_service = np.random.choice([0, 1, 2])  # 0=no, 1=DSL, 2=fiber
        online_security = np.random.choice([0, 1])

        # Churn probability depends on features (realistic patterns)
        churn_prob = 0.5

        # Higher churn for short tenure
        if tenure_months < 6:
            churn_prob += 0.3
        elif tenure_months < 12:
            churn_prob += 0.1

        # Higher churn for month-to-month contracts
        if contract_type == 0:
            churn_prob += 0.2

        # Higher churn for higher charges
        if monthly_charges > 100:
            churn_prob += 0.15

        # Higher churn with more support calls
        churn_prob += support_calls * 0.05

        # Lower churn with online security
        if online_security == 1:
            churn_prob -= 0.1

        # Add some randomness
        churn_prob += np.random.normal(0, 0.1)
        churn_prob = np.clip(churn_prob, 0, 1)

        # Determine churn
        churned = int(np.random.rand() < churn_prob)

        data.append({
            'tenure_months': tenure_months,
            'monthly_charges': monthly_charges,
            'total_charges': total_charges,
            'support_calls': support_calls,
            'contract_type': contract_type,
            'payment_method': payment_method,
            'internet_service': internet_service,
            'online_security': online_security
        })

        labels.append(churned)

    return pd.DataFrame(data), pd.Series(labels)


def visualize_results(opm, decision_maker, X_test, y_test, y_pred_proba):
    """Create comprehensive visualizations."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Feature importance
    feature_rankings = opm.rank_features()
    features, mi_values = zip(*feature_rankings)

    axes[0, 0].barh(features, mi_values, color='steelblue')
    axes[0, 0].set_xlabel('Mutual Information (bits)')
    axes[0, 0].set_title('Feature Importance for Churn Prediction')
    axes[0, 0].invert_yaxis()

    # Plot 2: Churn probability distribution
    churn_probs = y_pred_proba['P(Churn)']
    axes[0, 1].hist(churn_probs, bins=20, color='coral', edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(0.5, color='red', linestyle='--', label='Decision threshold')
    axes[0, 1].set_xlabel('P(Churn)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Distribution of Churn Probability')
    axes[0, 1].legend()

    # Plot 3: Recommended actions
    actions = []
    for prob in churn_probs:
        action, _ = decision_maker.choose_best_action(prob)
        actions.append(action)

    action_counts = pd.Series(actions).value_counts()
    axes[1, 0].bar(range(len(action_counts)), action_counts.values, color='lightgreen', edgecolor='black')
    axes[1, 0].set_xticks(range(len(action_counts)))
    axes[1, 0].set_xticklabels(action_counts.index, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Recommended Actions for Test Set')

    # Plot 4: Expected utility by churn probability
    test_probs = np.linspace(0, 1, 100)
    utilities_by_action = {action: [] for action in decision_maker.action_costs.keys()}

    for prob in test_probs:
        for action in decision_maker.action_costs.keys():
            util = decision_maker.compute_expected_utility(prob, action)
            utilities_by_action[action].append(util)

    for action, utilities in utilities_by_action.items():
        axes[1, 1].plot(test_probs, utilities, label=action, linewidth=2)

    axes[1, 1].set_xlabel('Churn Probability')
    axes[1, 1].set_ylabel('Expected Utility ($)')
    axes[1, 1].set_title('Expected Utility of Each Action')
    axes[1, 1].axhline(0, color='black', linestyle='--', linewidth=0.5)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('project_ideas/03_customer_churn/churn_analysis_results.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved to: project_ideas/03_customer_churn/churn_analysis_results.png")


def main():
    """Run the customer churn prediction and decision-making example."""

    print("=" * 70)
    print("Customer Churn Prediction OPM with Decision Making")
    print("=" * 70)

    # Generate data
    print("\n1. Generating synthetic customer data...")
    X, y = generate_synthetic_customer_data(n_samples=1000)
    print(f"   Created {len(X)} customer records with {len(X.columns)} features")
    print(f"   Churn rate: {y.mean():.1%}")

    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Train OPM
    print("\n2. Training Optimal Predictor Machine...")
    opm = CustomerChurnOPM(discretize_bins=5)
    opm.fit(X_train, y_train)
    print("   Model trained successfully!")

    # Feature importance
    print("\n3. Feature Importance (Mutual Information):")
    feature_rankings = opm.rank_features()
    for feature, mi in feature_rankings:
        print(f"   {feature:20s}: {mi:.4f} bits")

    # Make predictions
    print("\n4. Making churn predictions...")
    y_pred_proba = opm.predict_proba(X_test)
    y_pred = opm.predict(X_test)

    accuracy = (y_pred == y_test).mean()
    print(f"   Accuracy: {accuracy:.2%}")

    # Decision making
    print("\n5. Optimal Retention Strategy (Decision Making):")
    decision_maker = ChurnDecisionMaker()

    print(f"\n   Action costs: {decision_maker.action_costs}")
    print(f"   Action effectiveness: {decision_maker.action_effectiveness}")
    print(f"   Customer lifetime value: ${decision_maker.retention_value}")

    # Example decision making
    print("\n6. Example Decision Making (First 3 Customers):")
    for i in range(min(3, len(X_test))):
        churn_prob = y_pred_proba.iloc[i]['P(Churn)']
        best_action, utilities = decision_maker.choose_best_action(churn_prob)

        print(f"\n   Customer {i+1}:")
        print(f"   P(Churn) = {churn_prob:.3f}")
        print(f"   Expected utilities:")
        for action, util in sorted(utilities.items(), key=lambda x: x[1], reverse=True):
            marker = " <-- BEST" if action == best_action else ""
            print(f"      {action:15s}: ${util:6.2f}{marker}")
        print(f"   Recommended action: {best_action.upper()}")
        print(f"   True outcome: {'CHURNED' if y_test.iloc[i] == 1 else 'STAYED'}")

    # Overall strategy statistics
    print("\n7. Overall Strategy Statistics:")
    total_utility = 0
    action_counts = {}

    for i, churn_prob in enumerate(y_pred_proba['P(Churn)']):
        best_action, utilities = decision_maker.choose_best_action(churn_prob)
        total_utility += utilities[best_action]
        action_counts[best_action] = action_counts.get(best_action, 0) + 1

    print(f"   Total expected utility: ${total_utility:,.2f}")
    print(f"   Average utility per customer: ${total_utility / len(X_test):.2f}")
    print(f"\n   Action distribution:")
    for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"      {action:15s}: {count:3d} customers ({count/len(X_test):.1%})")

    # Visualizations
    print("\n8. Creating visualizations...")
    visualize_results(opm, decision_maker, X_test, y_test, y_pred_proba)

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)

    return opm, decision_maker, X_test, y_test, y_pred_proba


if __name__ == "__main__":
    opm, decision_maker, X_test, y_test, y_pred_proba = main()
