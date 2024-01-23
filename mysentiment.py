from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pandas as pd
import time

# User test data
statements = [
    "cant wait til her date this weekend",
    "I feel great today",
    "I dont feel great today",
    "I dont feel good or bad today",
    "demonstrated exceptional dedication and commitment, going above and beyond to ensure seamless operations",
    "i am really feeling great today, but i don't know how the day will turn out"
]

# Load data from CSV file
file_path = 'data.csv'
df = pd.read_csv(file_path, sep=',',
                 encoding='ISO-8859-1',
                 dtype={'Sentiment': str, 'Text': str},
                 header=None,
                 names=["Sentiment", "Text"])

# Convert 'Sentiment' to numeric
df['Sentiment'] = pd.to_numeric(df['Sentiment'], errors='coerce')

# Drop rows with NaN values in the target variable 'Sentiment'
df = df.dropna(subset=['Sentiment'])

# Vectorize the entire dataset
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(df['Text'])

# Train the classifier on the entire dataset
classifier = MultinomialNB()
classifier.fit(X_tfidf, df['Sentiment'])

# Vectorize the new statements
new_statements_tfidf = vectorizer.transform(statements)

# Make predictions on the new statements
new_statements_predictions = classifier.predict(new_statements_tfidf)

# Print or analyze the predictions
for statement, prediction in zip(statements, new_statements_predictions):
    print(f"Statement: {statement}")
    print(f"Predicted Sentiment: {prediction}\n")

# Display results
print(df[['Text', 'Sentiment']])

num_rows = df.shape[0]
print("Number of rows in the DataFrame:", num_rows)

# Count the occurrences of each sentiment label
sentiment_counts = df['Sentiment'].value_counts()

# Display the counts
print("Sentiment Counts:")
print(sentiment_counts)

print("4 means positive sentiment, 0 means negative sentiment")
