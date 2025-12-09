"""
GUI Application for Restaurant Review Analysis System.
Provides interfaces for both customers and restaurant owners.
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from datetime import datetime

from database import Database
from sentiment_analyzer import SentimentAnalyzer


class RestaurantReviewApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Restaurant Review Analysis System")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)

        # Initialize components
        self.db = Database()
        self.analyzer = SentimentAnalyzer()

        # Configure style
        self.style = ttk.Style()
        self.style.configure('Title.TLabel', font=('Helvetica', 16, 'bold'))
        self.style.configure('Header.TLabel', font=('Helvetica', 12, 'bold'))
        self.style.configure('Positive.TLabel', foreground='green')
        self.style.configure('Negative.TLabel', foreground='red')
        self.style.configure('Neutral.TLabel', foreground='orange')

        # Create main notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create tabs
        self.customer_frame = ttk.Frame(self.notebook)
        self.owner_frame = ttk.Frame(self.notebook)

        self.notebook.add(self.customer_frame, text="Customer - Submit Review")
        self.notebook.add(self.owner_frame, text="Owner - Analytics Dashboard")

        # Build interfaces
        self._build_customer_interface()
        self._build_owner_interface()

    def _build_customer_interface(self):
        """Build the customer review submission interface."""
        # Main container with padding
        main_frame = ttk.Frame(self.customer_frame, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="Submit Your Review",
            style='Title.TLabel'
        )
        title_label.pack(pady=(0, 20))

        # Customer name
        name_frame = ttk.Frame(main_frame)
        name_frame.pack(fill=tk.X, pady=5)

        ttk.Label(name_frame, text="Your Name (optional):").pack(side=tk.LEFT)
        self.customer_name_var = tk.StringVar()
        self.customer_name_entry = ttk.Entry(
            name_frame,
            textvariable=self.customer_name_var,
            width=30
        )
        self.customer_name_entry.pack(side=tk.LEFT, padx=(10, 0))

        # Food item selection
        food_frame = ttk.Frame(main_frame)
        food_frame.pack(fill=tk.X, pady=10)

        ttk.Label(food_frame, text="Select Food Item:").pack(side=tk.LEFT)

        self.food_items = self.db.get_all_food_items()
        self.food_item_var = tk.StringVar()

        self.food_combo = ttk.Combobox(
            food_frame,
            textvariable=self.food_item_var,
            values=[item[1] for item in self.food_items],
            state="readonly",
            width=30
        )
        self.food_combo.pack(side=tk.LEFT, padx=(10, 0))
        if self.food_items:
            self.food_combo.current(0)

        # Rating selection
        rating_frame = ttk.Frame(main_frame)
        rating_frame.pack(fill=tk.X, pady=10)

        ttk.Label(rating_frame, text="Rating (1-5):").pack(side=tk.LEFT)

        self.rating_var = tk.IntVar(value=3)
        for i in range(1, 6):
            ttk.Radiobutton(
                rating_frame,
                text=str(i),
                variable=self.rating_var,
                value=i
            ).pack(side=tk.LEFT, padx=5)

        # Review text
        review_label = ttk.Label(
            main_frame,
            text="Write your review:",
            style='Header.TLabel'
        )
        review_label.pack(anchor=tk.W, pady=(10, 5))

        self.review_text = scrolledtext.ScrolledText(
            main_frame,
            wrap=tk.WORD,
            width=70,
            height=8,
            font=('Helvetica', 10)
        )
        self.review_text.pack(fill=tk.X, pady=5)

        # Submit button
        submit_btn = ttk.Button(
            main_frame,
            text="Submit Review",
            command=self._submit_review
        )
        submit_btn.pack(pady=20)

        # Sentiment preview frame
        self.preview_frame = ttk.LabelFrame(
            main_frame,
            text="Sentiment Preview",
            padding="10"
        )
        self.preview_frame.pack(fill=tk.X, pady=10)

        self.sentiment_label = ttk.Label(
            self.preview_frame,
            text="Start typing to see sentiment analysis...",
            font=('Helvetica', 10)
        )
        self.sentiment_label.pack()

        # Bind text change event for live preview
        self.review_text.bind('<KeyRelease>', self._update_sentiment_preview)

    def _update_sentiment_preview(self, event=None):
        """Update sentiment preview as user types."""
        text = self.review_text.get("1.0", tk.END).strip()

        if len(text) < 10:
            self.sentiment_label.config(
                text="Keep typing for sentiment analysis...",
                style='TLabel'
            )
            return

        try:
            label, score, confidence = self.analyzer.predict_with_score(text)

            style_map = {
                'Positive': 'Positive.TLabel',
                'Negative': 'Negative.TLabel',
                'Neutral': 'Neutral.TLabel'
            }

            self.sentiment_label.config(
                text=f"Detected Sentiment: {label} (Score: {score:.2f}, Confidence: {confidence:.0%})",
                style=style_map.get(label, 'TLabel')
            )
        except Exception as e:
            self.sentiment_label.config(text=f"Analysis error: {str(e)}")

    def _submit_review(self):
        """Handle review submission."""
        review_text = self.review_text.get("1.0", tk.END).strip()

        if not review_text:
            messagebox.showerror("Error", "Please write a review before submitting.")
            return

        if len(review_text) < 10:
            messagebox.showerror("Error", "Review is too short. Please write at least 10 characters.")
            return

        # Get selected food item ID
        selected_idx = self.food_combo.current()
        if selected_idx < 0:
            messagebox.showerror("Error", "Please select a food item.")
            return

        food_item_id = self.food_items[selected_idx][0]

        # Analyze sentiment
        label, score, confidence = self.analyzer.predict_with_score(review_text)

        # Save to database
        customer_name = self.customer_name_var.get().strip() or "Anonymous"
        rating = self.rating_var.get()

        try:
            self.db.add_review(
                food_item_id=food_item_id,
                review_text=review_text,
                sentiment_score=score,
                sentiment_label=label,
                rating=rating,
                customer_name=customer_name
            )

            messagebox.showinfo(
                "Success",
                f"Thank you for your review!\n\n"
                f"Detected sentiment: {label}\n"
                f"Your feedback helps us improve!"
            )

            # Clear form
            self.review_text.delete("1.0", tk.END)
            self.customer_name_var.set("")
            self.rating_var.set(3)
            self.sentiment_label.config(
                text="Start typing to see sentiment analysis...",
                style='TLabel'
            )

            # Refresh owner dashboard if visible
            self._refresh_owner_dashboard()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save review: {str(e)}")

    def _build_owner_interface(self):
        """Build the owner analytics dashboard."""
        # Main container
        main_frame = ttk.Frame(self.owner_frame, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="Analytics Dashboard",
            style='Title.TLabel'
        )
        title_label.pack(pady=(0, 10))

        # Overall stats frame
        stats_frame = ttk.LabelFrame(main_frame, text="Overall Statistics", padding="10")
        stats_frame.pack(fill=tk.X, pady=5)

        self.stats_labels = {}
        stats_grid = ttk.Frame(stats_frame)
        stats_grid.pack(fill=tk.X)

        stat_items = [
            ('total_reviews', 'Total Reviews:'),
            ('avg_sentiment', 'Avg. Sentiment Score:'),
            ('positive_count', 'Positive Reviews:'),
            ('neutral_count', 'Neutral Reviews:'),
            ('negative_count', 'Negative Reviews:')
        ]

        for i, (key, label_text) in enumerate(stat_items):
            ttk.Label(stats_grid, text=label_text).grid(row=i // 3, column=(i % 3) * 2, sticky=tk.W, padx=5, pady=2)
            self.stats_labels[key] = ttk.Label(stats_grid, text="0", font=('Helvetica', 10, 'bold'))
            self.stats_labels[key].grid(row=i // 3, column=(i % 3) * 2 + 1, sticky=tk.W, padx=(0, 20), pady=2)

        # Least positive items frame
        least_positive_frame = ttk.LabelFrame(
            main_frame,
            text="Items Needing Improvement (Lowest Rated)",
            padding="10"
        )
        least_positive_frame.pack(fill=tk.X, pady=10)

        # Treeview for least positive items
        columns = ('rank', 'item', 'category', 'reviews', 'sentiment')
        self.least_positive_tree = ttk.Treeview(
            least_positive_frame,
            columns=columns,
            show='headings',
            height=5
        )

        self.least_positive_tree.heading('rank', text='#')
        self.least_positive_tree.heading('item', text='Food Item')
        self.least_positive_tree.heading('category', text='Category')
        self.least_positive_tree.heading('reviews', text='Reviews')
        self.least_positive_tree.heading('sentiment', text='Avg. Sentiment')

        self.least_positive_tree.column('rank', width=40, anchor=tk.CENTER)
        self.least_positive_tree.column('item', width=200)
        self.least_positive_tree.column('category', width=100)
        self.least_positive_tree.column('reviews', width=80, anchor=tk.CENTER)
        self.least_positive_tree.column('sentiment', width=100, anchor=tk.CENTER)

        self.least_positive_tree.pack(fill=tk.X, pady=5)

        # All food items analytics frame
        all_items_frame = ttk.LabelFrame(
            main_frame,
            text="All Food Items Analytics",
            padding="10"
        )
        all_items_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Treeview for all items
        columns = ('item', 'category', 'reviews', 'positive', 'neutral', 'negative', 'sentiment')
        self.all_items_tree = ttk.Treeview(
            all_items_frame,
            columns=columns,
            show='headings',
            height=8
        )

        self.all_items_tree.heading('item', text='Food Item')
        self.all_items_tree.heading('category', text='Category')
        self.all_items_tree.heading('reviews', text='Total')
        self.all_items_tree.heading('positive', text='Positive')
        self.all_items_tree.heading('neutral', text='Neutral')
        self.all_items_tree.heading('negative', text='Negative')
        self.all_items_tree.heading('sentiment', text='Avg. Score')

        self.all_items_tree.column('item', width=150)
        self.all_items_tree.column('category', width=100)
        self.all_items_tree.column('reviews', width=60, anchor=tk.CENTER)
        self.all_items_tree.column('positive', width=60, anchor=tk.CENTER)
        self.all_items_tree.column('neutral', width=60, anchor=tk.CENTER)
        self.all_items_tree.column('negative', width=60, anchor=tk.CENTER)
        self.all_items_tree.column('sentiment', width=80, anchor=tk.CENTER)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(all_items_frame, orient=tk.VERTICAL, command=self.all_items_tree.yview)
        self.all_items_tree.configure(yscrollcommand=scrollbar.set)

        self.all_items_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Bind double-click to show item reviews
        self.all_items_tree.bind('<Double-1>', self._show_item_reviews)

        # Recent reviews frame
        reviews_frame = ttk.LabelFrame(
            main_frame,
            text="Recent Reviews (Double-click item above to filter)",
            padding="10"
        )
        reviews_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Treeview for reviews
        columns = ('date', 'customer', 'item', 'sentiment', 'review')
        self.reviews_tree = ttk.Treeview(
            reviews_frame,
            columns=columns,
            show='headings',
            height=6
        )

        self.reviews_tree.heading('date', text='Date')
        self.reviews_tree.heading('customer', text='Customer')
        self.reviews_tree.heading('item', text='Food Item')
        self.reviews_tree.heading('sentiment', text='Sentiment')
        self.reviews_tree.heading('review', text='Review')

        self.reviews_tree.column('date', width=100)
        self.reviews_tree.column('customer', width=100)
        self.reviews_tree.column('item', width=120)
        self.reviews_tree.column('sentiment', width=80, anchor=tk.CENTER)
        self.reviews_tree.column('review', width=300)

        reviews_scrollbar = ttk.Scrollbar(reviews_frame, orient=tk.VERTICAL, command=self.reviews_tree.yview)
        self.reviews_tree.configure(yscrollcommand=reviews_scrollbar.set)

        self.reviews_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        reviews_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Refresh button
        refresh_btn = ttk.Button(
            main_frame,
            text="Refresh Dashboard",
            command=self._refresh_owner_dashboard
        )
        refresh_btn.pack(pady=10)

        # Initial load
        self._refresh_owner_dashboard()

    def _refresh_owner_dashboard(self):
        """Refresh all data in the owner dashboard."""
        # Update overall stats
        stats = self.db.get_overall_stats()
        self.stats_labels['total_reviews'].config(text=str(stats['total_reviews']))
        self.stats_labels['avg_sentiment'].config(text=f"{stats['avg_sentiment']:.3f}")
        self.stats_labels['positive_count'].config(text=str(stats['positive_count']))
        self.stats_labels['neutral_count'].config(text=str(stats['neutral_count']))
        self.stats_labels['negative_count'].config(text=str(stats['negative_count']))

        # Update least positive items
        self.least_positive_tree.delete(*self.least_positive_tree.get_children())
        least_positive = self.db.get_least_positive_items(5)
        for i, item in enumerate(least_positive, 1):
            sentiment_str = f"{item['avg_sentiment']:.3f}" if item['avg_sentiment'] else "N/A"
            self.least_positive_tree.insert('', 'end', values=(
                i,
                item['name'],
                item['category'] or 'N/A',
                item['review_count'],
                sentiment_str
            ))

        # Update all items analytics
        self.all_items_tree.delete(*self.all_items_tree.get_children())
        all_analytics = self.db.get_food_item_analytics()
        for item in all_analytics:
            sentiment_str = f"{item['avg_sentiment']:.3f}" if item['avg_sentiment'] else "N/A"
            self.all_items_tree.insert('', 'end', iid=item['id'], values=(
                item['name'],
                item['category'] or 'N/A',
                item['review_count'],
                item['positive_count'] or 0,
                item['neutral_count'] or 0,
                item['negative_count'] or 0,
                sentiment_str
            ))

        # Update recent reviews
        self._load_all_reviews()

    def _load_all_reviews(self):
        """Load all reviews into the reviews treeview."""
        self.reviews_tree.delete(*self.reviews_tree.get_children())
        reviews = self.db.get_all_reviews()

        for review in reviews[:50]:  # Show latest 50
            date_str = review['created_at'][:10] if review['created_at'] else 'N/A'
            review_preview = review['review_text'][:50] + "..." if len(review['review_text']) > 50 else review['review_text']

            self.reviews_tree.insert('', 'end', values=(
                date_str,
                review['customer_name'] or 'Anonymous',
                review['food_item'],
                review['sentiment_label'],
                review_preview
            ))

    def _show_item_reviews(self, event):
        """Show reviews for a specific food item."""
        selection = self.all_items_tree.selection()
        if not selection:
            return

        item_id = int(selection[0])
        reviews = self.db.get_reviews_for_item(item_id)

        # Update reviews treeview
        self.reviews_tree.delete(*self.reviews_tree.get_children())

        # Get item name for display
        item_name = self.all_items_tree.item(selection[0])['values'][0]

        for review in reviews:
            date_str = review['created_at'][:10] if review['created_at'] else 'N/A'
            review_preview = review['review_text'][:50] + "..." if len(review['review_text']) > 50 else review['review_text']

            self.reviews_tree.insert('', 'end', values=(
                date_str,
                review['customer_name'] or 'Anonymous',
                item_name,
                review['sentiment_label'],
                review_preview
            ))


def main():
    root = tk.Tk()
    app = RestaurantReviewApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
