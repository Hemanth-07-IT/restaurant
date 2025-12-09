"""
Sentiment Analysis module for Restaurant Review Analysis System.
Uses Scikit-learn for classification.
"""

import pickle
import os
from typing import Tuple, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from nlp_processor import NLPProcessor


class SentimentAnalyzer:
    def __init__(self, model_path: str = "sentiment_model.pkl"):
        self.model_path = model_path
        self.nlp_processor = NLPProcessor()
        self.model = None
        self.vectorizer = None

        # Load existing model or train a new one
        if os.path.exists(model_path):
            self.load_model()
        else:
            self._train_default_model()

    def _get_training_data(self) -> Tuple[List[str], List[str]]:
        """Get training data for sentiment analysis with Indian food reviews."""

        # Positive reviews - General
        positive_general = [
            "The food was absolutely delicious and amazing",
            "Best restaurant ever, highly recommend",
            "Excellent service and wonderful atmosphere",
            "Staff was very friendly and helpful",
            "Great value for money, portions were generous",
            "Lovely ambiance and great food quality",
            "Outstanding meal, everything was fresh",
            "Fantastic dining experience overall",
            "Superb quality ingredients and presentation",
            "Really enjoyed the meal, very satisfying",
            "The chef did an amazing job",
            "Wonderful flavors and great presentation",
            "Top notch service and delicious food",
            "Impressed by the quality and taste",
            "Love this place, always consistent",
            "Friendly staff made us feel welcome",
            "Exceeded expectations in every way",
            "Mouth-watering dishes and great service",
            "Beautiful presentation and delicious taste",
            "Can't wait to visit again soon",
            "Five star dining experience for sure",
            "Highly recommended to everyone",
            "Will definitely come back again",
            "Perfect blend of taste and quality",
            "Absolutely loved every bite",
        ]

        # Positive reviews - South Indian specific
        positive_south_indian = [
            "The masala dosa was crispy and perfectly spiced",
            "Best dosa I have ever tasted, so crispy",
            "The sambar was flavorful and had perfect consistency",
            "Idli was soft like cotton, melted in mouth",
            "Coconut chutney was fresh and delicious",
            "Medu vada was crispy outside and soft inside",
            "The filter coffee was authentic and aromatic",
            "Rava dosa was thin and crispy, loved it",
            "Uttapam was perfectly cooked with fresh toppings",
            "Rasam was tangy and soothing, perfect taste",
            "Pongal was creamy and well seasoned with ghee",
            "Upma was light and flavorful, great breakfast",
            "Curd rice was cooling and perfectly balanced",
            "Lemon rice had the perfect tanginess",
            "The dosa batter was fermented perfectly",
            "Authentic South Indian flavors, reminded me of home",
            "The gunpowder chutney was spicy and addictive",
            "Sambar had the perfect tamarind tang",
            "Idli sambar combo was heavenly",
            "The ghee roast dosa was outstanding",
            "Crispy vada with sambar is a must try here",
            "South Indian thali was complete and satisfying",
            "The coconut based curries were amazing",
            "Perfectly fermented idlis, so fluffy",
            "Filter kaapi was strong and aromatic",
        ]

        # Positive reviews - North Indian specific
        positive_north_indian = [
            "Butter chicken was rich and creamy, absolutely delicious",
            "The naan was soft and fresh from tandoor",
            "Dal makhani was slow cooked to perfection",
            "Paneer butter masala had the perfect gravy",
            "Biryani was aromatic with perfectly cooked rice",
            "Chole bhature was authentic and filling",
            "Tandoori chicken was juicy and well marinated",
            "The roti was soft and fresh, perfectly puffed",
            "Palak paneer had fresh spinach and soft paneer",
            "Rajma chawal reminded me of home cooked food",
            "Aloo paratha was crispy and stuffed generously",
            "The gravy had perfect blend of spices",
            "Chicken biryani had tender meat and fragrant rice",
            "Malai kofta was creamy and rich",
            "Kadai paneer had nice smoky flavor",
            "Shahi paneer was royal and delicious",
            "Jeera rice was fragrant and well cooked",
            "The tikka masala was perfectly spiced",
            "Garlic naan was soft with perfect garlic flavor",
            "Veg biryani had nice mix of vegetables",
            "The korma was mild and creamy, kids loved it",
            "Lassi was thick and refreshing",
            "Gulab jamun was soft and soaked in syrup perfectly",
            "Kheer was creamy with right amount of sweetness",
            "Masala chai was perfectly spiced and warming",
            "Jalebi was crispy and sweet, fresh made",
            "Rasmalai was soft and delicious",
            "The kebabs were succulent and flavorful",
            "Mango lassi was thick and fruity",
            "Raita was cool and refreshing with the biryani",
        ]

        # Negative reviews - General
        negative_general = [
            "The food was terrible and cold",
            "Worst restaurant experience ever",
            "Service was slow and rude staff",
            "Very disappointed with the quality",
            "Not worth the money at all",
            "Dirty tables and poor hygiene",
            "Food made me sick, never again",
            "Horrible experience from start to finish",
            "Poor quality ingredients used",
            "Did not enjoy the meal at all",
            "Bland flavors and bad presentation",
            "Terrible service and bad food",
            "Noisy and uncomfortable seating",
            "Very disappointed with everything",
            "Will never come back here",
            "Overpriced for such poor quality",
            "Rude staff ruined the evening",
            "Failed to meet basic expectations",
            "Unappetizing dishes and slow service",
            "Waste of time and money",
            "Zero stars if I could give it",
            "Food was stale and tasteless",
            "Unhygienic kitchen, saw cockroach",
            "Portions were too small for the price",
            "Had to wait forever for our order",
        ]

        # Negative reviews - South Indian specific
        negative_south_indian = [
            "Dosa was soggy and not crispy at all",
            "The sambar was watery and tasteless",
            "Idli was hard and stale, not fresh",
            "Coconut chutney tasted sour and old",
            "Medu vada was oily and undercooked inside",
            "Filter coffee was too weak and watery",
            "Rava dosa was thick and rubbery",
            "Uttapam was burnt and toppings were stale",
            "Rasam had no flavor, just like hot water",
            "Pongal was dry and lacked ghee",
            "Upma was lumpy and bland",
            "Curd rice was sour and unpleasant",
            "Lemon rice was too sour and oily",
            "The dosa was undercooked and pale",
            "Not authentic South Indian at all",
            "Sambar had no vegetables, very disappointing",
            "The batter was not fermented properly",
            "Chutney was too spicy, inedible",
            "Vada was stale and reheated",
            "The filter coffee was cold when served",
            "Idli was rubbery and hard to chew",
            "South Indian food here is not genuine",
            "Dosa fell apart, very poorly made",
            "No taste in any of the dishes",
            "The worst South Indian food I have had",
        ]

        # Negative reviews - North Indian specific
        negative_north_indian = [
            "Butter chicken was too oily and greasy",
            "The naan was hard and stale",
            "Dal makhani tasted like plain dal",
            "Paneer was rubbery and tasteless",
            "Biryani rice was overcooked and mushy",
            "Chole bhature was cold and stale",
            "Tandoori chicken was dry and burnt",
            "Roti was thick and undercooked",
            "Palak paneer had no spinach flavor",
            "Rajma was hard and undercooked",
            "Aloo paratha was oily and soggy",
            "The gravy was too spicy and burned my mouth",
            "Chicken biryani had very little chicken",
            "Malai kofta was too sweet and weird",
            "Kadai paneer had no kadai flavor",
            "Shahi paneer was bland and watery",
            "Jeera rice was undercooked and hard",
            "The tikka was charred and bitter",
            "Garlic naan had barely any garlic",
            "Veg biryani was just colored rice",
            "Lassi was too thin and watery",
            "Gulab jamun was hard as rock",
            "Kheer was too runny and tasteless",
            "Chai was cold and had no masala",
            "Jalebi was chewy and not crispy",
            "Rasmalai was sour and spoiled",
            "The kebabs were dry and overcooked",
            "Mango lassi had artificial flavor",
            "Raita was sour and curdled",
            "North Indian food here is terrible",
        ]

        # Neutral reviews - General
        neutral_general = [
            "The food was okay nothing special",
            "Average restaurant with decent options",
            "Service was neither good nor bad",
            "It was fine for a quick meal",
            "Standard quality nothing outstanding",
            "Clean but nothing remarkable",
            "Food was mediocre at best",
            "An ordinary dining experience overall",
            "Average ingredients and presentation",
            "The meal was passable but forgettable",
            "Nothing impressive about the cooking",
            "Basic flavors and simple presentation",
            "Adequate service and average food",
            "Typical restaurant nothing noteworthy",
            "Met expectations but no more",
            "Might come back might not",
            "Fair prices for fair quality",
            "Staff was indifferent but polite",
            "Just an average experience overall",
            "Unremarkable dishes and standard service",
            "Plain presentation and basic taste",
            "An ordinary meal overall",
            "Three out of five stars maybe",
            "Not bad but not great either",
            "Could be better could be worse",
        ]

        # Neutral reviews - South Indian specific
        neutral_south_indian = [
            "Dosa was okay, nothing extraordinary",
            "Sambar was decent but not memorable",
            "Idli was standard, like any other place",
            "Coconut chutney was average",
            "Medu vada was acceptable but plain",
            "Filter coffee was okay, not strong enough",
            "Rava dosa was fine but ordinary",
            "Uttapam was basic nothing special about it",
            "Rasam was passable but not flavorful",
            "Pongal was edible but lacked richness",
            "Upma was simple and plain",
            "Curd rice was just rice with curd",
            "Lemon rice was okay for the price",
            "Standard South Indian fare nothing more",
            "The dosa could have been crispier",
            "Sambar needed more vegetables",
            "Average breakfast nothing to write about",
            "Chutney was okay but not fresh",
            "Not the best but not terrible",
            "South Indian food was acceptable",
        ]

        # Neutral reviews - North Indian specific
        neutral_north_indian = [
            "Butter chicken was okay, expected more",
            "Naan was average, nothing special",
            "Dal makhani was decent but ordinary",
            "Paneer dishes were standard quality",
            "Biryani was okay but not aromatic",
            "Chole bhature was filling but average",
            "Tandoori items were passable",
            "Roti was okay but not soft enough",
            "Palak paneer was basic but edible",
            "Rajma was decent comfort food",
            "Paratha was okay but oily",
            "The curries were standard North Indian",
            "Biryani was acceptable nothing great",
            "Malai kofta was okay but too rich",
            "The food was filling but forgettable",
            "Lassi was okay, nothing special",
            "Desserts were average quality",
            "Chai was decent but could be better",
            "Standard North Indian restaurant food",
            "Not bad for the price we paid",
        ]

        # Combine all reviews
        positive_reviews = positive_general + positive_south_indian + positive_north_indian
        negative_reviews = negative_general + negative_south_indian + negative_north_indian
        neutral_reviews = neutral_general + neutral_south_indian + neutral_north_indian

        reviews = positive_reviews + negative_reviews + neutral_reviews
        labels = (['Positive'] * len(positive_reviews) +
                  ['Negative'] * len(negative_reviews) +
                  ['Neutral'] * len(neutral_reviews))

        return reviews, labels

    def _train_default_model(self):
        """Train the default sentiment model."""
        reviews, labels = self._get_training_data()

        # Preprocess reviews
        processed_reviews = [
            self.nlp_processor.preprocess_to_string(review)
            for review in reviews
        ]

        # Create and train pipeline
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ])

        self.model.fit(processed_reviews, labels)
        self.save_model()

    def train(self, reviews: List[str], labels: List[str],
              test_size: float = 0.2) -> dict:
        """
        Train the sentiment model with custom data.

        Args:
            reviews: List of review texts
            labels: List of sentiment labels ('Positive', 'Negative', 'Neutral')
            test_size: Fraction of data to use for testing

        Returns:
            Dictionary with training metrics
        """
        # Preprocess reviews
        processed_reviews = [
            self.nlp_processor.preprocess_to_string(review)
            for review in reviews
        ]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            processed_reviews, labels, test_size=test_size, random_state=42
        )

        # Create and train pipeline
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ])

        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        self.save_model()

        return {
            'accuracy': accuracy,
            'report': classification_report(y_test, y_pred)
        }

    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict sentiment of a single text.

        Args:
            text: Input review text

        Returns:
            Tuple of (sentiment_label, confidence_score)
        """
        if self.model is None:
            self._train_default_model()

        # Preprocess
        processed = self.nlp_processor.preprocess_to_string(text)

        # Predict
        label = self.model.predict([processed])[0]
        probabilities = self.model.predict_proba([processed])[0]
        confidence = max(probabilities)

        return label, confidence

    def predict_with_score(self, text: str) -> Tuple[str, float, float]:
        """
        Predict sentiment with a normalized score.

        Args:
            text: Input review text

        Returns:
            Tuple of (sentiment_label, sentiment_score, confidence)
            sentiment_score is between -1 (very negative) and 1 (very positive)
        """
        if self.model is None:
            self._train_default_model()

        # Preprocess
        processed = self.nlp_processor.preprocess_to_string(text)

        # Predict
        label = self.model.predict([processed])[0]
        probabilities = self.model.predict_proba([processed])[0]

        # Get class order
        classes = self.model.classes_

        # Calculate sentiment score
        neg_idx = np.where(classes == 'Negative')[0][0]
        neu_idx = np.where(classes == 'Neutral')[0][0]
        pos_idx = np.where(classes == 'Positive')[0][0]

        # Weighted score: -1 for negative, 0 for neutral, 1 for positive
        sentiment_score = (
            probabilities[pos_idx] * 1 +
            probabilities[neu_idx] * 0 +
            probabilities[neg_idx] * -1
        )

        confidence = max(probabilities)

        return label, sentiment_score, confidence

    def save_model(self):
        """Save the trained model to disk."""
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self):
        """Load the trained model from disk."""
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)


# Example usage and testing
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()

    test_reviews = [
        # Positive
        "The masala dosa was crispy and the sambar was delicious!",
        "Butter chicken here is amazing, best I've ever had",
        "Loved the filter coffee, very authentic taste",
        # Negative
        "The idli was hard and the chutney was sour, terrible",
        "Biryani was overcooked and had no flavor at all",
        "Worst paneer butter masala, too oily and bland",
        # Neutral
        "The dosa was okay, nothing special about it",
        "Dal makhani was decent but I've had better",
        "Average food, not bad but not great either",
    ]

    print("Sentiment Analysis Demo - Indian Food Reviews")
    print("=" * 60)

    for review in test_reviews:
        label, score, confidence = analyzer.predict_with_score(review)
        print(f"\nReview: {review}")
        print(f"Sentiment: {label} (Score: {score:.3f}, Confidence: {confidence:.3f})")
