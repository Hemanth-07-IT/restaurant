# Restaurant Review Analysis System

An end-to-end sentiment analysis system for restaurant reviews using Python, NLP, and machine learning.

## Features

- **Sentiment Analysis**: Analyzes customer reviews using NLP (tokenization, stop word removal, stemming) and classifies sentiment as Positive, Negative, or Neutral
- **Customer Interface**: Submit reviews for specific food items with real-time sentiment preview
- **Owner Dashboard**: View analytics including:
  - Overall statistics (total reviews, sentiment distribution)
  - Least positive-rated items for targeted improvement
  - Per-item analytics with review counts and sentiment scores
  - Recent reviews with filtering by food item

## Tech Stack

- **Python 3.8+**
- **NLTK**: Natural Language Processing (tokenization, stop words, stemming)
- **Scikit-learn**: Machine Learning (TF-IDF vectorization, Logistic Regression classifier)
- **Pandas**: Data manipulation
- **Tkinter**: GUI interface
- **SQLite**: Database storage

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd restaurant
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python main.py
   ```

## Project Structure

```
restaurant/
├── main.py              # Main entry point
├── gui_app.py           # Tkinter GUI application
├── database.py          # SQLite database operations
├── nlp_processor.py     # NLP preprocessing (tokenization, stemming)
├── sentiment_analyzer.py # Sentiment analysis model
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Usage

### Customer View
1. Select a food item from the dropdown
2. Write your review in the text area
3. See real-time sentiment analysis as you type
4. Optionally provide your name and rating
5. Click "Submit Review"

### Owner View
1. Switch to the "Owner - Analytics Dashboard" tab
2. View overall statistics at the top
3. See items needing improvement (lowest rated)
4. Browse all food items with their analytics
5. Double-click any item to see its specific reviews
6. Click "Refresh Dashboard" to update data

## How It Works

### NLP Preprocessing
1. **Text Cleaning**: Lowercase conversion, URL/email removal, number removal
2. **Tokenization**: Splitting text into individual words
3. **Stop Word Removal**: Removing common words (the, is, at, etc.)
4. **Stemming**: Reducing words to their root form (Porter Stemmer)

### Sentiment Classification
- Uses TF-IDF vectorization with unigrams and bigrams
- Logistic Regression classifier trained on restaurant review data
- Outputs sentiment label (Positive/Negative/Neutral) and confidence score
- Sentiment score ranges from -1 (very negative) to +1 (very positive)

## Database Schema

### food_items
- `id`: Primary key
- `name`: Food item name
- `category`: Category (Pizza, Pasta, etc.)
- `created_at`: Timestamp

### reviews
- `id`: Primary key
- `customer_name`: Optional customer name
- `food_item_id`: Foreign key to food_items
- `review_text`: The review content
- `sentiment_score`: Numerical sentiment (-1 to 1)
- `sentiment_label`: Classification (Positive/Negative/Neutral)
- `rating`: Optional 1-5 rating
- `created_at`: Timestamp
