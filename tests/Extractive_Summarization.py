import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter

# NLTK downloads are still needed if not already present
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

class ExtractiveSummarizer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def _preprocess_sentence(self, text):
        """Cleans, tokenizes, and lemmatizes a single sentence."""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = word_tokenize(text)
        return [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]

    def summarize(self, text, num_sentences=3):
        """
        Generates an extractive summary of the text.

        Args:
            text (str): The input text to be summarized.
            num_sentences (int): The desired number of sentences in the summary.

        Returns:
            str: The generated summary.
        """
        original_sentences = sent_tokenize(text)

        if len(original_sentences) <= num_sentences:
            return text

        # 1. Preprocess all sentences and calculate word frequencies for the whole document
        all_words = self._preprocess_sentence(text)
        word_frequencies = Counter(all_words)

        # 2. Score sentences based on word frequencies, normalized by sentence length
        sentence_scores = {}
        for i, sentence in enumerate(original_sentences):
            processed_words = self._preprocess_sentence(sentence)
            if not processed_words:
                continue # Skip empty sentences

            score = sum(word_frequencies.get(word, 0) for word in processed_words)
            # Normalize by the number of words in the processed sentence to avoid bias for long sentences
            sentence_scores[i] = score / len(processed_words)

        # 3. Get the indices of the top N sentences
        # The key is the original index of the sentence
        sorted_sentence_indices = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
        top_sentence_indices = sorted(sorted_sentence_indices[:num_sentences])

        # 4. Build the summary by joining the top sentences in their original order
        summary = ' '.join(original_sentences[i] for i in top_sentence_indices)
        return summary

# --- Usage ---
news_article = """
Artificial intelligence has revolutionized numerous industries in recent years, transforming how businesses operate and people work.
Machine learning algorithms now power everything from recommendation systems on streaming platforms to autonomous vehicles navigating city streets.
Companies like Google, Microsoft, and OpenAI have invested billions of dollars in developing advanced AI systems that can understand and generate human language.
The rise of large language models has enabled chatbots to engage in sophisticated conversations, assist with creative writing, and even help with programming tasks.
However, concerns about AI safety, job displacement, and ethical implications continue to grow among researchers and policymakers.
Many experts argue that proper regulation and guidelines are essential to ensure AI development benefits humanity while minimizing potential risks.
The future of artificial intelligence remains uncertain, but its impact on society will undoubtedly continue to expand in the coming decades.
"""

summarizer = ExtractiveSummarizer()
summary = summarizer.summarize(news_article, num_sentences=3)
print("--- Refactored Summary (3 sentences) ---")
print(summary)
