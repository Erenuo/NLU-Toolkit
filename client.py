import webbrowser
import requests
import json
import os

## Server and Test Data Settings ##
SERVER_URL = "http://127.0.0.1:8000"
ABSTRACTIVE_SUMMARIZE_ENDPOINT = f"{SERVER_URL}/summarize-text/abstractive"
EXTRACTIVE_SUMMARIZE_ENDPOINT = f"{SERVER_URL}/summarize-text/extractive"
TEXTRANK_SUMMARIZE_ENDPOINT = f"{SERVER_URL}/summarize-text/textrank"
NER_VISUALIZE_ENDPOINT = f"{SERVER_URL}/visualize-entities"
MORPHOLOGY_ENDPOINT = f"{SERVER_URL}/analyze-morphology"
NER_EXTRACT_ENDPOINT = f"{SERVER_URL}/extract-entities"
SENTIMENT_ENDPOINT = f"{SERVER_URL}/analyze-sentiment"
WORD_PROCESS_ENDPOINT = f"{SERVER_URL}/process-words"
TOKENIZER_ENDPOINT = f"{SERVER_URL}/tokenize-text"

## Texts and words to be tested ##
NEWS_ARTICLE_FOR_NER = "Apple Inc. is reportedly in talks to acquire a U.K. startup for $500 million. Tim Cook mentioned this in California."
POSITIVE_REVIEW = "This movie was absolutely fantastic! The acting was superb and the plot was thrilling."
WORDS_FOR_MORPHOLOGY = ["unhappiness", "restarted", "running", "quickly", "friendship", "beautiful"]
TEXT_FOR_TOKENIZATION = "Natural Language Processing (NLP) is exciting! Let's test this."
WORDS_TO_PROCESS = ["running", "children", "better", "flies", "studies", "happiness"]
LONG_TEXT_FOR_SUMMARY = """
Artificial intelligence has revolutionized numerous industries in recent years, transforming how businesses operate and people work.
Machine learning algorithms now power everything from recommendation systems on streaming platforms to autonomous vehicles navigating city streets.
Companies like Google, Microsoft, and OpenAI have invested billions of dollars in developing advanced AI systems that can understand and generate human language.
The rise of large language models has enabled chatbots to engage in sophisticated conversations, assist with creative writing, and even help with programming tasks.
However, concerns about AI safety, job displacement, and ethical implications continue to grow among researchers and policymakers.
"""

def run_client():
    """Sends requests to all endpoints on the server and processes the results."""
    print("--- Ultimate NLU API Client Started ---")
    
    try:
        health_check = requests.get(SERVER_URL)
        health_check.raise_for_status()
        print(f"✅ Server connection successful: {health_check.json().get('message')}\n")
    except requests.exceptions.RequestException as e:
        print(f"❌ ERROR: Could not connect to the server. Please make sure the server is running.\nDetail: {e}")
        return

    ## Calls all endpoints for testing ##
    test_endpoint("Summarization (Abstractive)", ABSTRACTIVE_SUMMARIZE_ENDPOINT, json_payload={"text": LONG_TEXT_FOR_SUMMARY, "max_length": 60, "min_length": 25})
    test_endpoint("Summarization (Extractive)", EXTRACTIVE_SUMMARIZE_ENDPOINT, json_payload={"text": LONG_TEXT_FOR_SUMMARY, "num_sentences": 2})
    test_endpoint("Summarization (TextRank)", TEXTRANK_SUMMARIZE_ENDPOINT, json_payload={"text": LONG_TEXT_FOR_SUMMARY, "num_sentences": 2})
    test_endpoint("Word Processing (Stem/Lemma)", WORD_PROCESS_ENDPOINT, json_payload={"words": WORDS_TO_PROCESS})
    test_endpoint("Entity Recognition (JSON)", NER_EXTRACT_ENDPOINT, json_payload={"text": NEWS_ARTICLE_FOR_NER})
    test_endpoint("Morphological Analysis", MORPHOLOGY_ENDPOINT, json_payload={"words": WORDS_FOR_MORPHOLOGY})
    test_endpoint("Tokenization", TOKENIZER_ENDPOINT, json_payload={"text": TEXT_FOR_TOKENIZATION})
    test_endpoint("Sentiment Analysis", SENTIMENT_ENDPOINT, json_payload={"text": POSITIVE_REVIEW})
    test_visualization()

def test_endpoint(test_name, url, json_payload):
    """Sends a POST request to the specified endpoint and prints the result."""
    print(f"--- Test: {test_name} ({url}) ---")
    try:
        response = requests.post(url, json=json_payload)
        response.raise_for_status()
        print("Server Response:\n" + json.dumps(response.json(), indent=2, ensure_ascii=False))
    except requests.exceptions.RequestException as e:
        print(f"❌ ERROR: Request failed: {e}")
    print("-" * 50 + "\n")

def test_visualization():
    """Tests the visualization endpoint and opens the result in a browser."""
    print(f"--- Test: Entity Recognition - Visualization ({NER_VISUALIZE_ENDPOINT}) ---")
    try:
        response = requests.post(NER_VISUALIZE_ENDPOINT, json={"text": NEWS_ARTICLE_FOR_NER})
        response.raise_for_status()
        
        output_filename = "ner_visualization_from_api.html"
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(response.text)
        
        print(f"✅ Visualization HTML saved to '{output_filename}'.")
        
        file_path = os.path.abspath(output_filename)
        webbrowser.open(f"file://{file_path}")
        print("Opening file in browser...")
    except requests.exceptions.RequestException as e:
        print(f"❌ ERROR: Request failed: {e}")
    print("-" * 50 + "\n")

if __name__ == "__main__":
    run_client()
