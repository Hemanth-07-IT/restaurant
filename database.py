"""
Database module for Restaurant Review Analysis System.
Handles SQLite database operations for reviews and food items.
"""

import sqlite3
from datetime import datetime
from typing import List, Tuple, Optional


class Database:
    def __init__(self, db_path: str = "restaurant_reviews.db"):
        self.db_path = db_path
        self.init_database()

    def get_connection(self) -> sqlite3.Connection:
        """Create and return a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_database(self):
        """Initialize database with required tables."""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Create food_items table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS food_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                category TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create reviews table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_name TEXT,
                food_item_id INTEGER,
                review_text TEXT NOT NULL,
                sentiment_score REAL,
                sentiment_label TEXT,
                rating INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (food_item_id) REFERENCES food_items (id)
            )
        ''')

        # Insert default food items if table is empty
        cursor.execute("SELECT COUNT(*) FROM food_items")
        if cursor.fetchone()[0] == 0:
            default_items = [
                # South Indian Dishes
                ("Masala Dosa", "South Indian"),
                ("Plain Dosa", "South Indian"),
                ("Rava Dosa", "South Indian"),
                ("Idli", "South Indian"),
                ("Medu Vada", "South Indian"),
                ("Uttapam", "South Indian"),
                ("Sambar", "South Indian"),
                ("Rasam", "South Indian"),
                ("Pongal", "South Indian"),
                ("Upma", "South Indian"),
                ("Curd Rice", "South Indian"),
                ("Lemon Rice", "South Indian"),
                ("Coconut Chutney", "South Indian"),
                ("Filter Coffee", "South Indian"),
                # North Indian Dishes
                ("Butter Chicken", "North Indian"),
                ("Paneer Butter Masala", "North Indian"),
                ("Dal Makhani", "North Indian"),
                ("Chole Bhature", "North Indian"),
                ("Rajma Chawal", "North Indian"),
                ("Aloo Paratha", "North Indian"),
                ("Naan", "North Indian"),
                ("Roti", "North Indian"),
                ("Tandoori Chicken", "North Indian"),
                ("Chicken Biryani", "North Indian"),
                ("Veg Biryani", "North Indian"),
                ("Palak Paneer", "North Indian"),
                ("Malai Kofta", "North Indian"),
                ("Kadai Paneer", "North Indian"),
                ("Shahi Paneer", "North Indian"),
                ("Jeera Rice", "North Indian"),
                ("Raita", "North Indian"),
                ("Gulab Jamun", "North Indian Desserts"),
                ("Jalebi", "North Indian Desserts"),
                ("Kheer", "North Indian Desserts"),
                ("Rasmalai", "North Indian Desserts"),
                ("Lassi", "Beverages"),
                ("Masala Chai", "Beverages"),
                ("Mango Lassi", "Beverages"),
            ]
            cursor.executemany(
                "INSERT INTO food_items (name, category) VALUES (?, ?)",
                default_items
            )

        conn.commit()
        conn.close()

    def add_food_item(self, name: str, category: str = None) -> int:
        """Add a new food item to the database."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO food_items (name, category) VALUES (?, ?)",
            (name, category)
        )
        item_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return item_id

    def get_all_food_items(self) -> List[Tuple]:
        """Get all food items from the database."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, category FROM food_items ORDER BY category, name")
        items = cursor.fetchall()
        conn.close()
        return [(row['id'], row['name'], row['category']) for row in items]

    def add_review(self, food_item_id: int, review_text: str,
                   sentiment_score: float, sentiment_label: str,
                   rating: int = None, customer_name: str = None) -> int:
        """Add a new review to the database."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO reviews (food_item_id, review_text, sentiment_score,
                                sentiment_label, rating, customer_name)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (food_item_id, review_text, sentiment_score, sentiment_label,
              rating, customer_name))
        review_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return review_id

    def get_all_reviews(self) -> List[dict]:
        """Get all reviews with food item details."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT r.id, r.customer_name, f.name as food_item, r.review_text,
                   r.sentiment_score, r.sentiment_label, r.rating, r.created_at
            FROM reviews r
            JOIN food_items f ON r.food_item_id = f.id
            ORDER BY r.created_at DESC
        ''')
        reviews = cursor.fetchall()
        conn.close()
        return [dict(row) for row in reviews]

    def get_food_item_analytics(self) -> List[dict]:
        """Get analytics for each food item (average sentiment, review count)."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT f.id, f.name, f.category,
                   COUNT(r.id) as review_count,
                   AVG(r.sentiment_score) as avg_sentiment,
                   SUM(CASE WHEN r.sentiment_label = 'Positive' THEN 1 ELSE 0 END) as positive_count,
                   SUM(CASE WHEN r.sentiment_label = 'Negative' THEN 1 ELSE 0 END) as negative_count,
                   SUM(CASE WHEN r.sentiment_label = 'Neutral' THEN 1 ELSE 0 END) as neutral_count
            FROM food_items f
            LEFT JOIN reviews r ON f.id = r.food_item_id
            GROUP BY f.id, f.name, f.category
            ORDER BY avg_sentiment ASC
        ''')
        analytics = cursor.fetchall()
        conn.close()
        return [dict(row) for row in analytics]

    def get_least_positive_items(self, limit: int = 5) -> List[dict]:
        """Get food items with lowest average sentiment scores."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT f.id, f.name, f.category,
                   COUNT(r.id) as review_count,
                   AVG(r.sentiment_score) as avg_sentiment
            FROM food_items f
            JOIN reviews r ON f.id = r.food_item_id
            GROUP BY f.id, f.name, f.category
            HAVING COUNT(r.id) > 0
            ORDER BY avg_sentiment ASC
            LIMIT ?
        ''', (limit,))
        items = cursor.fetchall()
        conn.close()
        return [dict(row) for row in items]

    def get_reviews_for_item(self, food_item_id: int) -> List[dict]:
        """Get all reviews for a specific food item."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT r.id, r.customer_name, r.review_text, r.sentiment_score,
                   r.sentiment_label, r.rating, r.created_at
            FROM reviews r
            WHERE r.food_item_id = ?
            ORDER BY r.created_at DESC
        ''', (food_item_id,))
        reviews = cursor.fetchall()
        conn.close()
        return [dict(row) for row in reviews]

    def get_overall_stats(self) -> dict:
        """Get overall statistics for the restaurant."""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM reviews")
        total_reviews = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(sentiment_score) FROM reviews")
        avg_sentiment = cursor.fetchone()[0] or 0

        cursor.execute('''
            SELECT sentiment_label, COUNT(*) as count
            FROM reviews
            GROUP BY sentiment_label
        ''')
        sentiment_dist = {row['sentiment_label']: row['count'] for row in cursor.fetchall()}

        conn.close()

        return {
            'total_reviews': total_reviews,
            'avg_sentiment': avg_sentiment,
            'positive_count': sentiment_dist.get('Positive', 0),
            'negative_count': sentiment_dist.get('Negative', 0),
            'neutral_count': sentiment_dist.get('Neutral', 0)
        }
