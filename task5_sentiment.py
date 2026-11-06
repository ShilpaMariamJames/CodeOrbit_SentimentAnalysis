import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline

# ---- Small labelled dataset (product reviews) ----
data = [
    ("I love this phone, the camera is amazing", "positive"),
    ("Excellent quality and fast delivery", "positive"),
    ("Absolutely fantastic, works perfectly", "positive"),
    ("Great value for money, very happy", "positive"),
    ("The battery life is awesome, I love it", "positive"),
    ("Best purchase I have made this year", "positive"),
    ("Very good product, highly recommend", "positive"),
    ("Super happy with the sound quality, great", "positive"),
    ("Wonderful design and excellent performance", "positive"),
    ("Good product, works great and looks nice", "positive"),
    ("I love the camera, amazing pictures", "positive"),
    ("Amazing product, I am very happy", "positive"),
    ("Love it, excellent and great quality", "positive"),
    ("Awesome speaker, I love the sound", "positive"),
    ("Really good and amazing value", "positive"),
    ("Terrible product, it broke after two days", "negative"),
    ("Worst purchase ever, complete waste of money", "negative"),
    ("Very bad quality, I am disappointed", "negative"),
    ("The screen stopped working, awful", "negative"),
    ("Poor battery and slow performance, bad", "negative"),
    ("I hate it, do not buy this", "negative"),
    ("Horrible customer service and broken item", "negative"),
    ("Disappointed, the product is useless", "negative"),
    ("Bad sound and cheap material, waste", "negative"),
    ("Awful experience, it broke quickly", "negative"),
    ("Useless charger, very disappointed", "negative"),
    ("Bad product, I hate the poor quality", "negative"),
    ("Disappointed with this useless phone", "negative"),
    ("Terrible battery, it is useless and bad", "negative"),
    ("Waste of money, awful and broken", "negative"),
    ("The product arrived on Tuesday", "neutral"),
    ("It is a phone with a black cover", "neutral"),
    ("The box contains a charger and a cable", "neutral"),
    ("I received the order today", "neutral"),
    ("The item is the same as shown in the picture", "neutral"),
    ("It comes in blue and red colours", "neutral"),
    ("The package was delivered by courier", "neutral"),
    ("The product weighs about 200 grams", "neutral"),
    ("It has two buttons on the side", "neutral"),
    ("The manual is written in English", "neutral"),
    ("The phone has a 6 inch screen", "neutral"),
    ("The order was shipped yesterday", "neutral"),
    ("It is available in two sizes", "neutral"),
    ("The battery is 5000 mAh", "neutral"),
    ("The cable is one metre long", "neutral"),
]
df = pd.DataFrame(data, columns=["text", "label"])

# ---- Split: 75% to train, 25% to test (stratify keeps all 3 labels in both) ----
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
)

# ---- Model: TF-IDF + Logistic Regression in one pipeline ----
model = make_pipeline(
    TfidfVectorizer(lowercase=True, ngram_range=(1, 2)),
    LogisticRegression(C=10, max_iter=1000),
)
model.fit(X_train, y_train)

# ---- Evaluate ----
predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)

print("=== Test set predictions ===")
for text, true, pred in zip(X_test, y_test, predictions):
    mark = "OK " if true == pred else "XX "
    print(f"{mark} [{pred:8}] (true: {true:8})  {text}")

print(f"\nAccuracy on test set: {acc:.2f}")
print("\nDetailed report:")
print(classification_report(y_test, predictions, zero_division=0))

# ---- Try new sentences ----
new_reviews = [
    "This laptop is amazing and super fast",
    "The headphones broke, really bad quality",
    "The parcel came in a brown box",
    "I am very happy with this purchase",
]
print("=== Predictions on new reviews ===")
for review, pred in zip(new_reviews, model.predict(new_reviews)):
    print(f"[{pred:8}]  {review}")

# ---- Short report ----
print(f"""
=== Short Report ===
Dataset : {len(df)} product reviews (15 positive, 15 negative, 15 neutral).
Model   : TF-IDF word features + Logistic Regression.
Result  : {acc:.0%} accuracy on {len(X_test)} unseen test reviews.
Findings:
 - Strong opinion words (love, excellent, broke, waste) are learned easily,
   so clear positive/negative reviews are classified well.
 - Mistakes happen when a review's key word (e.g. "awful", "best") appears
   only once or twice in the training data, so the model has not learned it yet.
 - The dataset is very small, so results can change with a different split.
   More data, or a pretrained model (e.g. VADER or a BERT model),
   would give better and more reliable results.
""")
