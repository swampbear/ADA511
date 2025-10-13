"""
Medical Diagnosis OPM - Optimal Predictor Machine for Disease Detection

This example demonstrates how OPM can be used for medical diagnosis by learning
conditional probability distributions from patient symptoms and test results.

Use case: Predicting diseases based on observed symptoms
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt


class MedicalDiagnosisOPM:
    """
    Optimal Predictor Machine for medical diagnosis.

    Uses Bayesian inference to predict diseases based on symptoms and test results.
    The OPM learns P(Disease | Symptoms) from training data.
    """

    def __init__(self):
        self.prior_disease = {}  # P(Disease)
        self.conditional_symptoms = {}  # P(Symptoms | Disease)
        self.diseases = []
        self.symptoms = []

    def fit(self, X, y):
        """
        Learn the conditional probability distributions from training data.

        Parameters:
        -----------
        X : DataFrame with symptom columns (binary: 0 or 1)
        y : Series with disease labels
        """
        self.diseases = y.unique()
        self.symptoms = X.columns.tolist()

        # Calculate prior P(Disease)
        total_samples = len(y)
        for disease in self.diseases:
            self.prior_disease[disease] = np.sum(y == disease) / total_samples

        # Calculate conditional P(Symptoms | Disease)
        self.conditional_symptoms = {}
        for disease in self.diseases:
            disease_mask = y == disease
            disease_samples = X[disease_mask]

            self.conditional_symptoms[disease] = {}
            for symptom in self.symptoms:
                # P(Symptom = 1 | Disease)
                prob_symptom = disease_samples[symptom].mean()
                self.conditional_symptoms[disease][symptom] = {
                    1: prob_symptom,
                    0: 1 - prob_symptom
                }

    def predict_proba(self, X):
        """
        Predict probability distribution over diseases for each patient.

        Uses Bayes' theorem: P(Disease | Symptoms) ∝ P(Symptoms | Disease) * P(Disease)

        Returns:
        --------
        DataFrame with probability for each disease
        """
        predictions = []

        for idx, patient in X.iterrows():
            disease_probs = {}

            for disease in self.diseases:
                # Start with prior
                prob = self.prior_disease[disease]

                # Multiply by likelihood of each symptom (assuming independence)
                for symptom in self.symptoms:
                    symptom_value = int(patient[symptom])
                    prob *= self.conditional_symptoms[disease][symptom][symptom_value]

                disease_probs[disease] = prob

            # Normalize to get proper probabilities
            total = sum(disease_probs.values())
            if total > 0:
                disease_probs = {k: v/total for k, v in disease_probs.items()}

            predictions.append(disease_probs)

        return pd.DataFrame(predictions)

    def predict(self, X):
        """Predict the most likely disease for each patient."""
        proba = self.predict_proba(X)
        return proba.idxmax(axis=1)

    def mutual_information(self, symptom):
        """
        Calculate mutual information between a symptom and disease.

        MI(Symptom; Disease) measures how much knowing the symptom reduces
        uncertainty about the disease.
        """
        mi = 0.0

        for disease in self.diseases:
            p_disease = self.prior_disease[disease]

            for symptom_val in [0, 1]:
                # P(Symptom, Disease)
                p_joint = self.conditional_symptoms[disease][symptom][symptom_val] * p_disease

                if p_joint > 0:
                    # P(Symptom)
                    p_symptom = sum(
                        self.conditional_symptoms[d][symptom][symptom_val] * self.prior_disease[d]
                        for d in self.diseases
                    )

                    if p_symptom > 0:
                        mi += p_joint * np.log2(p_joint / (p_symptom * p_disease))

        return mi

    def rank_symptoms(self):
        """Rank symptoms by their mutual information with disease."""
        mi_scores = {symptom: self.mutual_information(symptom) for symptom in self.symptoms}
        return sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)


def generate_synthetic_data(n_samples=1000):
    """
    Generate synthetic medical data for demonstration.

    Creates realistic correlations between symptoms and diseases.
    """
    np.random.seed(42)

    diseases = []
    data = []

    for _ in range(n_samples):
        # Randomly select a disease
        disease = np.random.choice(['Flu', 'COVID-19', 'Common Cold', 'Allergies'])
        diseases.append(disease)

        # Generate symptoms based on disease (with noise)
        if disease == 'Flu':
            symptoms = {
                'fever': np.random.rand() > 0.2,  # 80% have fever
                'cough': np.random.rand() > 0.3,  # 70% have cough
                'fatigue': np.random.rand() > 0.15,  # 85% have fatigue
                'runny_nose': np.random.rand() > 0.6,  # 40% have runny nose
                'shortness_of_breath': np.random.rand() > 0.8,  # 20% have SOB
                'loss_of_taste': np.random.rand() > 0.9,  # 10% have loss of taste
            }
        elif disease == 'COVID-19':
            symptoms = {
                'fever': np.random.rand() > 0.25,  # 75% have fever
                'cough': np.random.rand() > 0.2,  # 80% have cough
                'fatigue': np.random.rand() > 0.1,  # 90% have fatigue
                'runny_nose': np.random.rand() > 0.7,  # 30% have runny nose
                'shortness_of_breath': np.random.rand() > 0.4,  # 60% have SOB
                'loss_of_taste': np.random.rand() > 0.3,  # 70% have loss of taste
            }
        elif disease == 'Common Cold':
            symptoms = {
                'fever': np.random.rand() > 0.7,  # 30% have fever
                'cough': np.random.rand() > 0.4,  # 60% have cough
                'fatigue': np.random.rand() > 0.5,  # 50% have fatigue
                'runny_nose': np.random.rand() > 0.1,  # 90% have runny nose
                'shortness_of_breath': np.random.rand() > 0.95,  # 5% have SOB
                'loss_of_taste': np.random.rand() > 0.95,  # 5% have loss of taste
            }
        else:  # Allergies
            symptoms = {
                'fever': np.random.rand() > 0.95,  # 5% have fever
                'cough': np.random.rand() > 0.6,  # 40% have cough
                'fatigue': np.random.rand() > 0.6,  # 40% have fatigue
                'runny_nose': np.random.rand() > 0.05,  # 95% have runny nose
                'shortness_of_breath': np.random.rand() > 0.85,  # 15% have SOB
                'loss_of_taste': np.random.rand() > 0.98,  # 2% have loss of taste
            }

        data.append({k: int(v) for k, v in symptoms.items()})

    return pd.DataFrame(data), pd.Series(diseases)


def visualize_results(opm, X_test, y_test, y_pred_proba):
    """Create visualizations of the OPM predictions."""

    # 1. Symptom importance
    symptom_rankings = opm.rank_symptoms()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Mutual Information
    symptoms, mi_values = zip(*symptom_rankings)
    axes[0, 0].barh(symptoms, mi_values, color='steelblue')
    axes[0, 0].set_xlabel('Mutual Information (bits)')
    axes[0, 0].set_title('Symptom Importance for Diagnosis')
    axes[0, 0].invert_yaxis()

    # Plot 2: Prediction confidence distribution
    max_probs = y_pred_proba.max(axis=1)
    axes[0, 1].hist(max_probs, bins=20, color='coral', edgecolor='black')
    axes[0, 1].set_xlabel('Prediction Confidence')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Distribution of Prediction Confidence')
    axes[0, 1].axvline(max_probs.mean(), color='red', linestyle='--', label=f'Mean: {max_probs.mean():.2f}')
    axes[0, 1].legend()

    # Plot 3: Disease distribution in test set
    disease_counts = y_test.value_counts()
    axes[1, 0].bar(disease_counts.index, disease_counts.values, color='lightgreen', edgecolor='black')
    axes[1, 0].set_xlabel('Disease')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Disease Distribution in Test Set')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Plot 4: Example patient predictions
    sample_patients = y_pred_proba.iloc[:5]
    sample_patients.plot(kind='bar', ax=axes[1, 1], width=0.8)
    axes[1, 1].set_xlabel('Patient ID')
    axes[1, 1].set_ylabel('Probability')
    axes[1, 1].set_title('Disease Probability for First 5 Patients')
    axes[1, 1].legend(title='Disease', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig('project_ideas/01_medical_diagnosis/diagnosis_results.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved to: project_ideas/01_medical_diagnosis/diagnosis_results.png")


def main():
    """Run the medical diagnosis OPM example."""

    print("=" * 70)
    print("Medical Diagnosis OPM - Disease Prediction from Symptoms")
    print("=" * 70)

    # Generate synthetic data
    print("\n1. Generating synthetic patient data...")
    X, y = generate_synthetic_data(n_samples=1000)
    print(f"   Created {len(X)} patient records with {len(X.columns)} symptoms")
    print(f"   Diseases: {y.unique().tolist()}")

    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Train OPM
    print("\n2. Training Optimal Predictor Machine...")
    opm = MedicalDiagnosisOPM()
    opm.fit(X_train, y_train)
    print("   Model trained successfully!")

    # Display learned distributions
    print("\n3. Learned Prior Probabilities P(Disease):")
    for disease, prob in opm.prior_disease.items():
        print(f"   {disease:15s}: {prob:.3f}")

    # Symptom importance
    print("\n4. Symptom Importance (Mutual Information):")
    symptom_rankings = opm.rank_symptoms()
    for symptom, mi in symptom_rankings:
        print(f"   {symptom:20s}: {mi:.4f} bits")

    # Make predictions
    print("\n5. Making predictions on test set...")
    y_pred_proba = opm.predict_proba(X_test)
    y_pred = opm.predict(X_test)

    # Evaluate
    accuracy = (y_pred.values == y_test.values).mean()
    print(f"   Accuracy: {accuracy:.2%}")

    # Show example predictions
    print("\n6. Example Predictions (First 3 Patients):")
    for i in range(min(3, len(X_test))):
        print(f"\n   Patient {i+1}:")
        print(f"   Symptoms: {X_test.iloc[i].to_dict()}")
        print(f"   Predicted probabilities:")
        for disease in opm.diseases:
            prob = y_pred_proba.iloc[i][disease]
            print(f"      {disease:15s}: {prob:.3f} {'<-- Most likely' if disease == y_pred.iloc[i] else ''}")
        print(f"   True diagnosis: {y_test.iloc[i]}")
        print(f"   Prediction: {'CORRECT' if y_pred.iloc[i] == y_test.iloc[i] else 'INCORRECT'}")

    # Create visualizations
    print("\n7. Creating visualizations...")
    visualize_results(opm, X_test, y_test, y_pred_proba)

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)

    return opm, X_test, y_test, y_pred_proba


if __name__ == "__main__":
    opm, X_test, y_test, y_pred_proba = main()
