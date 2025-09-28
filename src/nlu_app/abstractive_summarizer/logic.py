from transformers import pipeline

summarizer_pipeline = None

def load_abstractive_model(model_name: str = "facebook/bart-large-cnn"):
    global summarizer_pipeline
    print(f"--- Loading Abstractive Summarizer Model ({model_name}) ---")
    try:
        # Initialize the pipeline for summarization
        summarizer_pipeline = pipeline("summarization", model=model_name)
        print("Abstractive summarizer model loaded successfully.")
    except Exception as e:
        print(f"Error loading Hugging Face model '{model_name}': {e}")
        # Re-raise the exception to stop the server from starting incorrectly.
        raise e

def generate_abstractive_summary(
    text: str, 
    max_length: int = 130, 
    min_length: int = 30
) -> str:
    """
    Args:
        text (str): The input text to be summarized.
        max_length (int): The maximum length of the summary.
        min_length (int): The minimum length of the summary.

    Returns:
        str: The generated summary text.
    """
    if summarizer_pipeline is None:
        raise RuntimeError("Abstractive summarizer model is not loaded. Please run 'load_abstractive_model' at startup.")


    summary_result = summarizer_pipeline(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary_result[0]['summary_text']
