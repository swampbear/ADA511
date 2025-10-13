"""
Spam Detection OPM - Optimal Predictor Machine for Email Classification

This example demonstrates using OPM with text data to classify emails as spam or ham.
Uses word frequency features and Bayesian inference (similar to Naive Bayes).

Use case: Classifying emails as spam or legitimate based on word occurrences
"""

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import re


class SpamDetectionOPM:
    """
    Optimal Predictor Machine for spam detection.

    Uses word frequency features and conditional probability distributions
    to classify text as spam or ham (legitimate email).
    """

    def __init__(self, vocabulary_size=100, smoothing=1.0):
        """
        Parameters:
        -----------
        vocabulary_size : int
            Number of most common words to include in vocabulary
        smoothing : float
            Laplace smoothing parameter to handle unseen words
        """
        self.vocabulary_size = vocabulary_size
        self.smoothing = smoothing
        self.vocabulary = []
        self.prior = {}  # P(Class)
        self.word_given_class = {}  # P(Word | Class)
        self.classes = ['spam', 'ham']

    def _tokenize(self, text):
        """Convert text to lowercase tokens (words)."""
        # Simple tokenization: lowercase, split on non-alphanumeric
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        return words

    def _build_vocabulary(self, texts):
        """Build vocabulary from most common words in corpus."""
        all_words = []
        for text in texts:
            all_words.extend(self._tokenize(text))

        # Get most common words
        word_counts = Counter(all_words)
        self.vocabulary = [word for word, count in word_counts.most_common(self.vocabulary_size)]

    def _text_to_features(self, text):
        """Convert text to feature vector (word counts)."""
        words = self._tokenize(text)
        word_counts = Counter(words)

        features = {}
        for word in self.vocabulary:
            features[word] = word_counts.get(word, 0)

        return features

    def fit(self, texts, labels):
        """
        Learn conditional probability distributions from training data.

        Parameters:
        -----------
        texts : list of strings
            Email text content
        labels : list of strings
            'spam' or 'ham' labels
        """
        # Build vocabulary
        self._build_vocabulary(texts)

        # Calculate prior P(Class)
        label_counts = Counter(labels)
        total = len(labels)
        for cls in self.classes:
            self.prior[cls] = label_counts.get(cls, 0) / total

        # Calculate P(Word | Class) with Laplace smoothing
        self.word_given_class = {cls: {} for cls in self.classes}

        for cls in self.classes:
            # Get all texts for this class
            class_texts = [texts[i] for i in range(len(texts)) if labels[i] == cls]

            # Count words in this class
            word_counts = Counter()
            for text in class_texts:
                words = self._tokenize(text)
                word_counts.update(words)

            # Total word count in this class
            total_words = sum(word_counts.values())

            # Calculate P(Word | Class) with smoothing
            for word in self.vocabulary:
                # Laplace smoothing: (count + smoothing) / (total + smoothing * vocab_size)
                count = word_counts.get(word, 0)
                prob = (count + self.smoothing) / (total_words + self.smoothing * len(self.vocabulary))
                self.word_given_class[cls][word] = prob

    def predict_proba(self, texts):
        """
        Predict probability distribution over classes for each text.

        Uses Bayes' theorem: P(Class | Words) ∝ P(Words | Class) * P(Class)

        Returns:
        --------
        DataFrame with probability for each class
        """
        predictions = []

        for text in texts:
            features = self._text_to_features(text)
            class_probs = {}

            for cls in self.classes:
                # Start with log prior to avoid numerical underflow
                log_prob = np.log(self.prior[cls])

                # Add log probabilities for each word (assuming independence)
                for word, count in features.items():
                    if word in self.word_given_class[cls]:
                        word_prob = self.word_given_class[cls][word]
                        log_prob += count * np.log(word_prob)

                class_probs[cls] = log_prob

            # Convert log probabilities back to probabilities (normalized)
            max_log_prob = max(class_probs.values())
            exp_probs = {cls: np.exp(prob - max_log_prob) for cls, prob in class_probs.items()}
            total = sum(exp_probs.values())
            class_probs = {cls: prob / total for cls, prob in exp_probs.items()}

            predictions.append(class_probs)

        return pd.DataFrame(predictions)

    def predict(self, texts):
        """Predict the most likely class for each text."""
        proba = self.predict_proba(texts)
        return proba.idxmax(axis=1).tolist()

    def get_most_informative_words(self, n=20):
        """
        Find words that are most indicative of spam vs ham.

        Uses likelihood ratio: P(Word | Spam) / P(Word | Ham)
        """
        word_ratios = {}

        for word in self.vocabulary:
            spam_prob = self.word_given_class['spam'][word]
            ham_prob = self.word_given_class['ham'][word]

            # Avoid division by zero
            if ham_prob > 0:
                ratio = spam_prob / ham_prob
                word_ratios[word] = ratio

        # Sort by distance from 1.0 (either very spam or very ham)
        sorted_words = sorted(word_ratios.items(), key=lambda x: abs(np.log(x[1])), reverse=True)

        return sorted_words[:n]


def generate_synthetic_emails(n_samples=500):
    """
    Generate synthetic email data for demonstration.

    Creates realistic spam and ham emails with characteristic words.
    """
    np.random.seed(42)

    # Spam keywords
    spam_words = [
        'free', 'win', 'winner', 'cash', 'prize', 'click', 'offer', 'buy',
        'discount', 'guarantee', 'urgent', 'act', 'now', 'limited', 'bonus',
        'credit', 'money', 'pharmacy', 'viagra', 'weight', 'loss'
    ]

    # Ham keywords
    ham_words = [
        'meeting', 'schedule', 'project', 'report', 'team', 'update', 'please',
        'thanks', 'attached', 'document', 'review', 'deadline', 'question',
        'help', 'regarding', 'discussion', 'feedback', 'conference', 'call'
    ]

    # Common neutral words
    common_words = [
        'the', 'is', 'at', 'which', 'on', 'a', 'an', 'as', 'are', 'was',
        'be', 'by', 'for', 'with', 'this', 'that', 'from', 'have', 'has'
    ]

    emails = []
    labels = []

    for _ in range(n_samples):
        # Randomly decide if spam or ham
        is_spam = np.random.rand() > 0.5

        if is_spam:
            # Generate spam email
            label = 'spam'

            # More spam words, some common words
            selected_spam = np.random.choice(spam_words, size=np.random.randint(5, 12), replace=True)
            selected_common = np.random.choice(common_words, size=np.random.randint(3, 8), replace=True)

            # Occasionally add a ham word (noise)
            if np.random.rand() > 0.7:
                selected_ham = np.random.choice(ham_words, size=np.random.randint(1, 3), replace=True)
                all_words = list(selected_spam) + list(selected_common) + list(selected_ham)
            else:
                all_words = list(selected_spam) + list(selected_common)

        else:
            # Generate ham email
            label = 'ham'

            # More ham words, some common words
            selected_ham = np.random.choice(ham_words, size=np.random.randint(5, 12), replace=True)
            selected_common = np.random.choice(common_words, size=np.random.randint(3, 8), replace=True)

            # Occasionally add a spam word (noise)
            if np.random.rand() > 0.8:
                selected_spam = np.random.choice(spam_words, size=np.random.randint(1, 2), replace=True)
                all_words = list(selected_ham) + list(selected_common) + list(selected_spam)
            else:
                all_words = list(selected_ham) + list(selected_common)

        # Shuffle words and create email text
        np.random.shuffle(all_words)
        email_text = ' '.join(all_words)

        emails.append(email_text)
        labels.append(label)

    return emails, labels


def visualize_results(opm, texts_test, labels_test, predictions_proba):
    """Create visualizations of spam detection results."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Most informative words
    informative_words = opm.get_most_informative_words(n=15)
    words, ratios = zip(*informative_words)
    colors = ['red' if ratio > 1 else 'green' for ratio in ratios]

    axes[0, 0].barh(range(len(words)), [np.log(r) for r in ratios], color=colors)
    axes[0, 0].set_yticks(range(len(words)))
    axes[0, 0].set_yticklabels(words)
    axes[0, 0].set_xlabel('Log Likelihood Ratio: log(P(Word|Spam) / P(Word|Ham))')
    axes[0, 0].set_title('Most Informative Words (Red=Spam, Green=Ham)')
    axes[0, 0].axvline(0, color='black', linestyle='--', linewidth=0.5)
    axes[0, 0].invert_yaxis()

    # Plot 2: Prediction confidence
    spam_probs = predictions_proba['spam']
    axes[0, 1].hist(spam_probs, bins=20, color='coral', edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('P(Spam)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Distribution of Spam Probability')
    axes[0, 1].axvline(0.5, color='red', linestyle='--', label='Decision threshold')
    axes[0, 1].legend()

    # Plot 3: Confusion matrix
    predictions = predictions_proba.idxmax(axis=1).tolist()
    true_labels = labels_test

    confusion = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    for true, pred in zip(true_labels, predictions):
        if true == 'spam' and pred == 'spam':
            confusion['TP'] += 1
        elif true == 'ham' and pred == 'ham':
            confusion['TN'] += 1
        elif true == 'ham' and pred == 'spam':
            confusion['FP'] += 1
        elif true == 'spam' and pred == 'ham':
            confusion['FN'] += 1

    conf_matrix = np.array([[confusion['TP'], confusion['FN']],
                            [confusion['FP'], confusion['TN']]])

    im = axes[1, 0].imshow(conf_matrix, cmap='Blues')
    axes[1, 0].set_xticks([0, 1])
    axes[1, 0].set_yticks([0, 1])
    axes[1, 0].set_xticklabels(['Spam', 'Ham'])
    axes[1, 0].set_yticklabels(['Spam', 'Ham'])
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('True')
    axes[1, 0].set_title('Confusion Matrix')

    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = axes[1, 0].text(j, i, conf_matrix[i, j],
                                   ha="center", va="center", color="black", fontsize=16)

    # Plot 4: Performance metrics
    accuracy = (confusion['TP'] + confusion['TN']) / sum(confusion.values())
    precision = confusion['TP'] / (confusion['TP'] + confusion['FP']) if (confusion['TP'] + confusion['FP']) > 0 else 0
    recall = confusion['TP'] / (confusion['TP'] + confusion['FN']) if (confusion['TP'] + confusion['FN']) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [accuracy, precision, recall, f1]

    axes[1, 1].bar(metrics, values, color=['steelblue', 'coral', 'lightgreen', 'gold'], edgecolor='black')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_ylim(0, 1.1)
    axes[1, 1].set_title('Performance Metrics')
    axes[1, 1].axhline(1.0, color='gray', linestyle='--', linewidth=0.5)

    # Add value labels on bars
    for i, v in enumerate(values):
        axes[1, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('project_ideas/02_spam_detection/spam_detection_results.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved to: project_ideas/02_spam_detection/spam_detection_results.png")


def main():
    """Run the spam detection OPM example."""

    print("=" * 70)
    print("Spam Detection OPM - Email Classification")
    print("=" * 70)

    # Generate synthetic data
    print("\n1. Generating synthetic email data...")
    emails, labels = generate_synthetic_emails(n_samples=500)
    print(f"   Created {len(emails)} emails")
    print(f"   Spam: {labels.count('spam')}, Ham: {labels.count('ham')}")

    # Split data
    split_idx = int(0.8 * len(emails))
    texts_train = emails[:split_idx]
    labels_train = labels[:split_idx]
    texts_test = emails[split_idx:]
    labels_test = labels[split_idx:]

    # Train OPM
    print("\n2. Training Optimal Predictor Machine...")
    opm = SpamDetectionOPM(vocabulary_size=50, smoothing=1.0)
    opm.fit(texts_train, labels_train)
    print(f"   Model trained successfully!")
    print(f"   Vocabulary size: {len(opm.vocabulary)} words")

    # Display learned distributions
    print("\n3. Learned Prior Probabilities:")
    for cls, prob in opm.prior.items():
        print(f"   P({cls:4s}) = {prob:.3f}")

    # Most informative words
    print("\n4. Most Informative Words for Spam Detection:")
    informative_words = opm.get_most_informative_words(n=10)
    print(f"   {'Word':15s} {'Spam/Ham Ratio':>15s} {'Indicator':>12s}")
    print("   " + "-" * 45)
    for word, ratio in informative_words:
        indicator = "SPAM" if ratio > 1 else "HAM"
        print(f"   {word:15s} {ratio:15.2f} {indicator:>12s}")

    # Make predictions
    print("\n5. Making predictions on test set...")
    predictions_proba = opm.predict_proba(texts_test)
    predictions = opm.predict(texts_test)

    # Evaluate
    accuracy = sum([1 for true, pred in zip(labels_test, predictions) if true == pred]) / len(labels_test)
    print(f"   Accuracy: {accuracy:.2%}")

    # Show example predictions
    print("\n6. Example Predictions (First 3 Emails):")
    for i in range(min(3, len(texts_test))):
        print(f"\n   Email {i+1}:")
        print(f"   Text: {texts_test[i][:80]}...")
        print(f"   P(Spam) = {predictions_proba.iloc[i]['spam']:.3f}")
        print(f"   P(Ham)  = {predictions_proba.iloc[i]['ham']:.3f}")
        print(f"   Prediction: {predictions[i].upper()}")
        print(f"   True label: {labels_test[i].upper()}")
        print(f"   Result: {'CORRECT' if predictions[i] == labels_test[i] else 'INCORRECT'}")

    # Create visualizations
    print("\n7. Creating visualizations...")
    visualize_results(opm, texts_test, labels_test, predictions_proba)

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)

    return opm, texts_test, labels_test, predictions_proba


if __name__ == "__main__":
    opm, texts_test, labels_test, predictions_proba = main()
