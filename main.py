"""
Restaurant Review Analysis System
Main entry point for the application.
"""

import sys
import os

# Ensure the current directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_dependencies():
    """Check if all required dependencies are installed."""
    missing = []

    try:
        import nltk
    except ImportError:
        missing.append('nltk')

    try:
        import sklearn
    except ImportError:
        missing.append('scikit-learn')

    try:
        import pandas
    except ImportError:
        missing.append('pandas')

    try:
        import numpy
    except ImportError:
        missing.append('numpy')

    if missing:
        print("Missing dependencies detected!")
        print("Please install the following packages:")
        print(f"  pip install {' '.join(missing)}")
        print("\nOr run: pip install -r requirements.txt")
        return False

    return True


def main():
    """Main entry point for the application."""
    print("=" * 50)
    print("Restaurant Review Analysis System")
    print("=" * 50)
    print()

    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    print("All dependencies OK!")
    print()

    # Download NLTK data if needed
    print("Initializing NLP components...")
    import nltk
    nltk_packages = ['punkt', 'stopwords', 'punkt_tab']
    for package in nltk_packages:
        try:
            if 'punkt' in package:
                nltk.data.find(f'tokenizers/{package}')
            else:
                nltk.data.find(f'corpora/{package}')
        except LookupError:
            print(f"  Downloading {package}...")
            nltk.download(package, quiet=True)
    print("NLP components ready!")
    print()

    # Initialize sentiment analyzer (will train if needed)
    print("Loading sentiment analysis model...")
    from sentiment_analyzer import SentimentAnalyzer
    analyzer = SentimentAnalyzer()
    print("Sentiment model ready!")
    print()

    # Initialize database
    print("Initializing database...")
    from database import Database
    db = Database()
    print("Database ready!")
    print()

    # Launch GUI
    print("Launching GUI application...")
    print("=" * 50)

    import tkinter as tk
    from gui_app import RestaurantReviewApp

    root = tk.Tk()
    app = RestaurantReviewApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
