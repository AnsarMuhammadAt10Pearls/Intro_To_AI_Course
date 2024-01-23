from transformers import pipeline

sentiment_analyzer = pipeline("sentiment-analysis")

statements = [
    "cant wait til her date this weekend",
    "I feel great today",
    "I dont feel great today",
    "I dont feel good or bad today",
    "demonstrated exceptional dedication and commitment, going above and beyond to ensure seamless operations",
    "i am really feeling great today, but i don't know how the day will turn out"
]

for statement in statements:
    result = sentiment_analyzer(statement)[0]
    sentiment = result['label']
    print(f"Statement: {statement}")
    print(f"Sentiment: {sentiment}\n")
