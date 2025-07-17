from typing import List, Dict, Optional
from spacy import displacy
import spacy

# --- Module-Level Variable ---
# This will hold the loaded spaCy model.
nlp_model = None

def load_ner_model(model_name: str = "en_core_web_sm"):
    """
    Loads the spaCy language model into memory.
    This should be called once when the application starts.
    
    Args:
        model_name (str): The name of the spaCy model to load.
    """
    global nlp_model
    print(f"--- Loading spaCy model '{model_name}' ---")
    try:
        nlp_model = spacy.load(model_name)
        print("spaCy model loaded successfully.")
    except OSError:
        error_msg = f"SpaCy model '{model_name}' not found. Please run 'python -m spacy download {model_name}'"
        print(error_msg)
        # Re-raise the exception to stop the server from starting incorrectly.
        raise OSError(error_msg)

def extract_named_entities(text: str, labels_to_include: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """
    Processes a text to find and extract named entities.

    Args:
        text (str): The input text to analyze.
        labels_to_include (Optional[List[str]]): A list of entity labels to filter for 
                                                  (e.g., ["PERSON", "ORG"]). If None, all entities are returned.

    Returns:
        List[Dict[str, str]]: A list of dictionaries, where each dictionary represents a found entity.
    """
    if nlp_model is None:
        raise RuntimeError("spaCy model is not loaded. Please run 'load_ner_model' at application startup.")

    doc = nlp_model(text)
    entities = []
    
    for ent in doc.ents:
        # If a filter is provided, only include entities with a matching label.
        if labels_to_include is None or ent.label_ in labels_to_include:
            entity_data = {
                "text": ent.text,
                "label": ent.label_,
                "explanation": spacy.explain(ent.label_)
            }
            entities.append(entity_data)
            
    return entities

def visualize_entities(text: str) -> str:
    """
    Generates an HTML string with highlighted named entities using displaCy.

    Args:
        text (str): The input text to visualize.

    Returns:
        str: A self-contained HTML string for rendering in a browser.
    """
    if nlp_model is None:
        raise RuntimeError("spaCy model is not loaded. Please run 'load_ner_model' at application startup.")

    doc = nlp_model(text)
    
    # The `page=True` argument creates a full HTML document.
    html = displacy.render(doc, style="ent", page=True)
    return html
