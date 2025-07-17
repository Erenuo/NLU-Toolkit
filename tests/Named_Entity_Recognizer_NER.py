import spacy
from spacy import displacy # For visualization


# Load the English language model
try:
    nlp = spacy.load("en_core_web_sm")
    print("spaCy model 'en_core_web_sm' loaded successfully.")
except OSError:
    print("SpaCy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm'")
    exit()

# News Article Text
news_article_text = """
Apple Inc. is reportedly in talks to acquire a U.K. based artificial intelligence startup called 'DeepSense' for $500 million.
The deal, if finalized, would bolster Apple's efforts in AI and machine learning, particularly for its Siri voice assistant.
The negotiations began in early July 2025 and are expected to conclude by the end of this month.
Tim Cook, Apple's CEO, mentioned the potential acquisition during a quarterly earnings call last Tuesday.
DeepSense, founded by Dr. Alice Smith, is headquartered in London and has about 150 employees.
This marks Apple's third acquisition in the AI sector this year, following purchases of smaller firms in California and New York.
The company's stock rose by 1.5% on the Nasdaq after the news broke.
"""

# Process the Text with spaCy
doc = nlp(news_article_text)

# Extract and Display Named Entities
print("\n--- Detected Named Entities ---")
if doc.ents:
    for ent in doc.ents:
        # ent.text: the recognized entity string
        # ent.label_: the category of the entity (e.g., ORG, GPE)
        # spacy.explain(ent.label_): a brief description of the label
        print(f"Text: '{ent.text}' | Label: {ent.label_} | Explanation: {spacy.explain(ent.label_)}")
else:
    print("No named entities found.")

# Visualize Entities
# This will generate an HTML file that highlights the entities in browser
html = displacy.render(doc, style="ent", page=True)
with open("ner_visualization.html", "w", encoding="utf-8") as f:
    f.write(html)
print("\n--- Visualization Saved ---")
print("Open 'ner_visualization.html' in your browser to see the entities highlighted.")

# You can also filter by specific entity types if you only care about certain ones
print("\n--- Filtering for specific entities (e.g., PERSON, ORG, GPE) ---")
for ent in doc.ents:
    if ent.label_ in ["PERSON", "DATE", "MONEY"]:
        print(f"Filtered: '{ent.text}' | Label: {ent.label_}")
