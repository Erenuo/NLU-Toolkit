import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import pandas as pd

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

# Initialize
porter = PorterStemmer()
lemmatizer = WordNetLemmatizer()

words = [
    "running", "children", "better", "flies", "mice", "studies", "leaves", 
    "driving", "boxes", "easier", "swimming", "feet", "countries", "worse",
    "organizing", "arguing", "analysis", "universal", "caring", "happiness"
]

# Comparison table
data = []
for word in words:
    stemmed = porter.stem(word)
    lemmatized = lemmatizer.lemmatize(word, get_wordnet_pos(word))
    data.append({
        'Original': word,
        'Stemmed': stemmed,
        'Lemmatized': lemmatized
    })

df = pd.DataFrame(data)
print(df.to_string(index=False))
