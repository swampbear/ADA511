"""
Sentiment Analysis OPM - Optimal Predictor Machine for Product Reviews

This example demonstrates using OPM for multi-class sentiment classification
of product reviews. Shows how to handle ordinal outcomes and compute
uncertainty in predictions.

Use case: Classifying product reviews into sentiment categories (very negative to very positive)
"""

import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import re


class SentimentAnalysisOPM:
    """
    Optimal Predictor Machine for sentiment analysis with ordinal classes.

    Classifies text into sentiment levels: 1 (very negative) to 5 (very positive)
    """

    def __init__(self, vocabulary_size=100, smoothing=1.0):
        """
        Parameters:
        -----------
        vocabulary_size : int
            Number of most common words to include in vocabulary
        smoothing : float
            Laplace smoothing parameter
        """
        self.vocabulary_size = vocabulary_size
        self.smoothing = smoothing
        self.vocabulary = []
        self.prior = {}  # P(Sentiment)
        self.word_given_sentiment = {}  # P(Word | Sentiment)
        self.sentiments = [1, 2, 3, 4, 5]  # 1=very negative, 5=very positive

    def _tokenize(self, text):
        """Convert text to lowercase tokens."""
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        return words

    def _build_vocabulary(self, texts):
        """Build vocabulary from most common words."""
        all_words = []
        for text in texts:
            all_words.extend(self._tokenize(text))

        word_counts = Counter(all_words)
        self.vocabulary = [word for word, count in word_counts.most_common(self.vocabulary_size)]

    def _text_to_features(self, text):
        """Convert text to word count features."""
        words = self._tokenize(text)
        word_counts = Counter(words)

        features = {}
        for word in self.vocabulary:
            features[word] = word_counts.get(word, 0)

        return features

    def fit(self, texts, sentiments):
        """
        Learn conditional probability distributions from training data.

        Parameters:
        -----------
        texts : list of strings
            Review texts
        sentiments : list of int
            Sentiment ratings (1-5)
        """
        # Build vocabulary
        self._build_vocabulary(texts)

        # Calculate prior P(Sentiment)
        sentiment_counts = Counter(sentiments)
        total = len(sentiments)
        for sentiment in self.sentiments:
            self.prior[sentiment] = sentiment_counts.get(sentiment, 0) / total

        # Calculate P(Word | Sentiment)
        self.word_given_sentiment = {s: {} for s in self.sentiments}

        for sentiment in self.sentiments:
            # Get all texts for this sentiment
            sentiment_texts = [texts[i] for i in range(len(texts)) if sentiments[i] == sentiment]

            # Count words
            word_counts = Counter()
            for text in sentiment_texts:
                words = self._tokenize(text)
                word_counts.update(words)

            total_words = sum(word_counts.values())

            # Calculate probabilities with smoothing
            for word in self.vocabulary:
                count = word_counts.get(word, 0)
                prob = (count + self.smoothing) / (total_words + self.smoothing * len(self.vocabulary))
                self.word_given_sentiment[sentiment][word] = prob

    def predict_proba(self, texts):
        """
        Predict probability distribution over sentiments for each text.

        Returns:
        --------
        DataFrame with probability for each sentiment level
        """
        predictions = []

        for text in texts:
            features = self._text_to_features(text)
            sentiment_probs = {}

            for sentiment in self.sentiments:
                # Start with log prior
                log_prob = np.log(self.prior[sentiment])

                # Add log probabilities for each word
                for word, count in features.items():
                    if word in self.word_given_sentiment[sentiment]:
                        word_prob = self.word_given_sentiment[sentiment][word]
                        log_prob += count * np.log(word_prob)

                sentiment_probs[sentiment] = log_prob

            # Convert to normalized probabilities
            max_log_prob = max(sentiment_probs.values())
            exp_probs = {s: np.exp(prob - max_log_prob) for s, prob in sentiment_probs.items()}
            total = sum(exp_probs.values())
            sentiment_probs = {s: prob / total for s, prob in exp_probs.items()}

            predictions.append(sentiment_probs)

        return pd.DataFrame(predictions)

    def predict(self, texts):
        """Predict the most likely sentiment for each text."""
        proba = self.predict_proba(texts)
        return proba.idxmax(axis=1).tolist()

    def predict_expected_rating(self, texts):
        """
        Predict expected rating as weighted average of sentiment probabilities.

        This gives a continuous estimate rather than discrete classification.
        """
        proba = self.predict_proba(texts)
        expected_ratings = []

        for idx in range(len(proba)):
            expected_rating = sum(
                sentiment * proba.iloc[idx][sentiment]
                for sentiment in self.sentiments
            )
            expected_ratings.append(expected_rating)

        return expected_ratings

    def get_sentiment_words(self, sentiment, n=15):
        """
        Find words most characteristic of a sentiment level.

        Returns words with highest P(Word | Sentiment) relative to overall frequency.
        """
        # Calculate P(Word) across all sentiments
        p_word = {}
        for word in self.vocabulary:
            p_word[word] = sum(
                self.word_given_sentiment[s][word] * self.prior[s]
                for s in self.sentiments
            )

        # Calculate likelihood ratio for this sentiment
        word_scores = {}
        for word in self.vocabulary:
            if p_word[word] > 0:
                ratio = self.word_given_sentiment[sentiment][word] / p_word[word]
                word_scores[word] = ratio

        # Sort and return top n
        sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_words[:n]


def generate_synthetic_reviews(n_samples=600):
    """
    Generate synthetic product reviews with varying sentiments.
    """
    np.random.seed(42)

    # Word lists for different sentiments
    very_negative_words = ['terrible', 'awful', 'horrible', 'worst', 'disgusting', 'useless', 'waste', 'broken', 'defective', 'disappointed']
    negative_words = ['bad', 'poor', 'disappointing', 'mediocre', 'unhappy', 'issue', 'problem', 'lacking', 'subpar']
    neutral_words = ['okay', 'average', 'fine', 'acceptable', 'decent', 'moderate', 'reasonable', 'standard']
    positive_words = ['good', 'nice', 'pleased', 'happy', 'satisfied', 'quality', 'solid', 'recommend', 'works']
    very_positive_words = ['excellent', 'amazing', 'outstanding', 'perfect', 'fantastic', 'love', 'best', 'incredible', 'awesome', 'superb']

    common_words = ['product', 'price', 'quality', 'delivery', 'ordered', 'received', 'used', 'bought', 'tried', 'the', 'is', 'was', 'very', 'really']

    reviews = []
    sentiments = []

    for _ in range(n_samples):
        # Randomly select sentiment (1-5)
        sentiment = np.random.choice([1, 2, 3, 4, 5])
        sentiments.append(sentiment)

        # Generate review based on sentiment
        if sentiment == 1:  # Very negative
            main_words = np.random.choice(very_negative_words, size=np.random.randint(4, 8), replace=True)
            secondary_words = np.random.choice(negative_words, size=np.random.randint(2, 4), replace=True)
        elif sentiment == 2:  # Negative
            main_words = np.random.choice(negative_words, size=np.random.randint(4, 8), replace=True)
            secondary_words = np.random.choice(very_negative_words + neutral_words, size=np.random.randint(2, 4), replace=True)
        elif sentiment == 3:  # Neutral
            main_words = np.random.choice(neutral_words, size=np.random.randint(4, 8), replace=True)
            secondary_words = np.random.choice(positive_words + negative_words, size=np.random.randint(2, 4), replace=True)
        elif sentiment == 4:  # Positive
            main_words = np.random.choice(positive_words, size=np.random.randint(4, 8), replace=True)
            secondary_words = np.random.choice(very_positive_words + neutral_words, size=np.random.randint(2, 4), replace=True)
        else:  # Very positive
            main_words = np.random.choice(very_positive_words, size=np.random.randint(4, 8), replace=True)
            secondary_words = np.random.choice(positive_words, size=np.random.randint(2, 4), replace=True)

        # Add common words
        common = np.random.choice(common_words, size=np.random.randint(5, 10), replace=True)

        # Combine and shuffle
        all_words = list(main_words) + list(secondary_words) + list(common)
        np.random.shuffle(all_words)

        review_text = ' '.join(all_words)
        reviews.append(review_text)

    return reviews, sentiments


def visualize_results(opm, texts_test, sentiments_test, predictions_proba):
    """Create comprehensive visualizations."""

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Plot 1: Sentiment distribution (true vs predicted)
    ax1 = fig.add_subplot(gs[0, :2])
    true_counts = Counter(sentiments_test)
    pred_sentiments = predictions_proba.idxmax(axis=1).tolist()
    pred_counts = Counter(pred_sentiments)

    x = np.arange(len(opm.sentiments))
    width = 0.35

    true_vals = [true_counts.get(s, 0) for s in opm.sentiments]
    pred_vals = [pred_counts.get(s, 0) for s in opm.sentiments]

    ax1.bar(x - width/2, true_vals, width, label='True', color='steelblue', edgecolor='black')
    ax1.bar(x + width/2, pred_vals, width, label='Predicted', color='coral', edgecolor='black')
    ax1.set_xlabel('Sentiment Rating')
    ax1.set_ylabel('Count')
    ax1.set_title('Sentiment Distribution: True vs Predicted')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['1\n(Very Neg)', '2\n(Neg)', '3\n(Neutral)', '4\n(Pos)', '5\n(Very Pos)'])
    ax1.legend()

    # Plot 2: Prediction confidence
    ax2 = fig.add_subplot(gs[0, 2])
    max_probs = predictions_proba.max(axis=1)
    ax2.hist(max_probs, bins=20, color='lightgreen', edgecolor='black')
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Count')
    ax2.set_title('Prediction Confidence')
    ax2.axvline(max_probs.mean(), color='red', linestyle='--', label=f'Mean: {max_probs.mean():.2f}')
    ax2.legend()

    # Plot 3-7: Characteristic words for each sentiment
    for i, sentiment in enumerate(opm.sentiments):
        row = (i // 3) + 1
        col = i % 3
        ax = fig.add_subplot(gs[row, col])

        sentiment_words = opm.get_sentiment_words(sentiment, n=10)
        if sentiment_words:
            words, scores = zip(*sentiment_words)
            colors = plt.cm.RdYlGn([(sentiment - 1) / 4] * len(words))

            ax.barh(range(len(words)), scores, color=colors, edgecolor='black')
            ax.set_yticks(range(len(words)))
            ax.set_yticklabels(words, fontsize=8)
            ax.set_xlabel('Likelihood Ratio', fontsize=8)
            ax.set_title(f'Sentiment {sentiment} Words', fontsize=9)
            ax.invert_yaxis()

    plt.savefig('project_ideas/04_sentiment_analysis/sentiment_analysis_results.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved to: project_ideas/04_sentiment_analysis/sentiment_analysis_results.png")


def main():
    """Run the sentiment analysis OPM example."""

    print("=" * 70)
    print("Sentiment Analysis OPM - Product Review Classification")
    print("=" * 70)

    # Generate data
    print("\n1. Generating synthetic product reviews...")
    reviews, sentiments = generate_synthetic_reviews(n_samples=600)
    print(f"   Created {len(reviews)} reviews")
    print(f"   Sentiment distribution: {dict(Counter(sentiments))}")

    # Split data
    split_idx = int(0.8 * len(reviews))
    texts_train = reviews[:split_idx]
    sentiments_train = sentiments[:split_idx]
    texts_test = reviews[split_idx:]
    sentiments_test = sentiments[split_idx:]

    # Train OPM
    print("\n2. Training Optimal Predictor Machine...")
    opm = SentimentAnalysisOPM(vocabulary_size=60, smoothing=1.0)
    opm.fit(texts_train, sentiments_train)
    print(f"   Model trained successfully!")
    print(f"   Vocabulary size: {len(opm.vocabulary)} words")

    # Display prior
    print("\n3. Learned Prior Probabilities P(Sentiment):")
    for sentiment in opm.sentiments:
        bar = '█' * int(opm.prior[sentiment] * 50)
        print(f"   Sentiment {sentiment}: {opm.prior[sentiment]:.3f} {bar}")

    # Characteristic words
    print("\n4. Most Characteristic Words by Sentiment:")
    for sentiment in [1, 3, 5]:  # Show extreme and neutral
        words = opm.get_sentiment_words(sentiment, n=8)
        sentiment_label = {1: "Very Negative", 3: "Neutral", 5: "Very Positive"}[sentiment]
        word_list = [w for w, _ in words]
        print(f"   {sentiment_label:15s}: {', '.join(word_list)}")

    # Make predictions
    print("\n5. Making predictions on test set...")
    predictions_proba = opm.predict_proba(texts_test)
    predictions = opm.predict(texts_test)
    expected_ratings = opm.predict_expected_rating(texts_test)

    # Evaluate
    accuracy = sum([1 for true, pred in zip(sentiments_test, predictions) if true == pred]) / len(sentiments_test)
    mae = np.mean([abs(true - pred) for true, pred in zip(sentiments_test, predictions)])

    print(f"   Accuracy (exact match): {accuracy:.2%}")
    print(f"   Mean Absolute Error: {mae:.3f} stars")

    # Evaluate with tolerance (within 1 star)
    close_accuracy = sum([1 for true, pred in zip(sentiments_test, predictions) if abs(true - pred) <= 1]) / len(sentiments_test)
    print(f"   Accuracy (within 1 star): {close_accuracy:.2%}")

    # Show examples
    print("\n6. Example Predictions (First 3 Reviews):")
    for i in range(min(3, len(texts_test))):
        print(f"\n   Review {i+1}:")
        print(f"   Text: {texts_test[i][:80]}...")
        print(f"   Probability distribution:")
        for sentiment in opm.sentiments:
            prob = predictions_proba.iloc[i][sentiment]
            bar = '█' * int(prob * 20)
            print(f"      Sentiment {sentiment}: {prob:.3f} {bar}")
        print(f"   Predicted sentiment: {predictions[i]}")
        print(f"   Expected rating: {expected_ratings[i]:.2f}")
        print(f"   True sentiment: {sentiments_test[i]}")
        print(f"   Result: {'CORRECT' if predictions[i] == sentiments_test[i] else 'INCORRECT'}")

    # Create visualizations
    print("\n7. Creating visualizations...")
    visualize_results(opm, texts_test, sentiments_test, predictions_proba)

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print("\nKey Insights:")
    print(f"  - OPM provides full probability distribution over all sentiment levels")
    print(f"  - Expected ratings ({np.mean(expected_ratings):.2f}) can be more informative than hard predictions")
    print(f"  - Characteristic words reveal what the model has learned about each sentiment")
    print(f"  - Confidence scores help identify uncertain predictions")

    return opm, texts_test, sentiments_test, predictions_proba


if __name__ == "__main__":
    opm, texts_test, sentiments_test, predictions_proba = main()
