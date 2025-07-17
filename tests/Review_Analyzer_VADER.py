from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
## nltk.download('vader_lexicon')

# VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool that is specifically 
# attuned to sentiments expressed in social media. It's part of NLTK.

# Create a VADER sentiment analyzer object
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment_vader(text):
    scores = analyzer.polarity_scores(text)
    return scores

# Example product reviews
reviews = [
    "This product is absolutely amazing! I love it.",
    "The customer service was terrible. Very disappointed.",
    "It's okay, nothing special. Works as advertised.",
    "Not bad at all. I might even buy it again.",
    "Worst experience ever!!! Avoid at all costs."
]

print("\n*** VADER Sentiment Analysis ***\n")

for review in reviews:
    sentiment_scores = analyze_sentiment_vader(review)
    compound_score = sentiment_scores['compound']

    sentiment = "Neutral"
    if compound_score >= 0.05:
        sentiment = "Positive"
    elif compound_score <= -0.05:
        sentiment = "Negative"

    print(f"Review: \"{review}\"")
    print(f"Scores: {sentiment_scores}")
    print(f"Overall Sentiment: {sentiment}\n")
