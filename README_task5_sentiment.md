# Sentiment Analysis on Product Reviews

AI Internship – Task 5

## About
This project classifies product reviews as **positive, negative or neutral** using a machine learning model trained on 45 labelled reviews (15 of each class).

## Approach
1. The data is split into 75% training and 25% testing.
2. **TF-IDF** (single words and two-word phrases) converts the text into numbers.
3. A **Logistic Regression** model learns which words belong to each sentiment.
4. The model is evaluated on unseen reviews and tested on new sentences.

## Tech used
Python, pandas, scikit-learn

## How to run
```
pip install pandas scikit-learn
python task5_sentiment.py
```

## Results
- **Test accuracy: 75%** (12 unseen reviews)
- Example predictions:
  - "This laptop is amazing and super fast" → positive
  - "The headphones broke, really bad quality" → negative
  - "The parcel came in a brown box" → neutral

## Findings
- Strong opinion words (love, excellent, broke, waste) are learned well.
- Mistakes happen when a key word appears only once or twice in the training data.
- The dataset is small, so more data or a pretrained model (VADER or BERT) would improve the results.
