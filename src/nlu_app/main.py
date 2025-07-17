from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from typing import List, Optional

from .named_entity_recognizer import load_ner_model, extract_named_entities, visualize_entities
from .abstractive_summarizer import load_abstractive_model, generate_abstractive_summary
from .extractive_summarizer import load_summarizer_tools, generate_extractive_summary
from .morphological_analyzer import analyze_word_list as analyze_morphology_list
from .word_processor import load_word_processing_tools, process_word_list
from .sentiment_analyzer import load_sentiment_model, predict_sentiment
from .textrank_summarizer import generate_textrank_summary
from .tokenizer.logic import tokenize_text

## Pydantic Models ##

class TextInput(BaseModel):
    text: str

class ExtractiveSummarizationInput(BaseModel):
    text: str
    num_sentences: int = Field(3, gt=0, description="Number of sentences for extractive methods.")

class AbstractiveSummarizationInput(BaseModel):
    text: str
    max_length: int = Field(130, gt=20, description="Max token length for abstractive summary.")
    min_length: int = Field(30, gt=0, description="Min token length for abstractive summary.")

class WordListInput(BaseModel):
    words: List[str]

class SentimentOutput(BaseModel):
    original_text: str
    predicted_sentiment: str

class NerEntity(BaseModel):
    text: str
    label: str
    explanation: str

class NerOutput(BaseModel):
    original_text: str
    entities: List[NerEntity]

class ProcessedWord(BaseModel):
    original: str
    stemmed: str
    lemmatized: str

class WordProcessingOutput(BaseModel):
    results: List[ProcessedWord]

class MorphologyResult(BaseModel):
    original_word: str
    prefix: Optional[str] = None
    root: str
    suffix: Optional[str] = None
    suffix_function: Optional[str] = None
    inferred_pos: str

class MorphologyOutput(BaseModel):
    results: List[MorphologyResult]

class SummaryOutput(BaseModel):
    original_text_length: int
    summary: str
    summary_length: int
    method: str

class TokenizerOutput(BaseModel):
    original_text: str
    tokens: List[str]
    token_count: int

## FastAPI Application ##

app = FastAPI(
    title="Ultimate NLU API",
    description="An API that performs NLU tasks such as sentiment analysis, named entity recognition, word processing, morphological analysis, tokenization, and text summarization using three different methods.",
    version="1.0.0"
)

## Server Startup ##

@app.on_event("startup")
def startup_event():
    """This function runs once when the server starts and loads all models/tools."""
    print("--- Starting Server: Loading All Models and Tools ---")
    try:
        load_sentiment_model(model_dir='saved_model', nltk_data_dir='nltk_data')
        load_ner_model()
        load_word_processing_tools()
        load_summarizer_tools()
        load_abstractive_model() # Load the new abstractive model
        print("\n--- All models and tools loaded successfully. Server is ready. ---")
    except (FileNotFoundError, OSError, Exception) as e:
        print(f"\nFATAL ERROR: An error occurred during loading. Server could not be started.")
        print(f"Error Detail: {e}")
        raise e

## API Endpoints ##

@app.get("/", tags=["Health Check"])
def read_root():
    return {"status": "ok", "message": "Welcome to the Ultimate NLU API!"}

@app.post("/summarize-text/extractive", response_model=SummaryOutput, tags=["Summarization"])
def api_summarize_text_extractive(payload: ExtractiveSummarizationInput):
    summary_text = generate_extractive_summary(payload.text, payload.num_sentences)
    return SummaryOutput(
        original_text_length=len(payload.text), summary=summary_text,
        summary_length=len(summary_text), method="Frequency-Based Extractive"
    )

@app.post("/summarize-text/textrank", response_model=SummaryOutput, tags=["Summarization"])
def api_summarize_text_textrank(payload: ExtractiveSummarizationInput):
    summary_text = generate_textrank_summary(payload.text, payload.num_sentences)
    return SummaryOutput(
        original_text_length=len(payload.text), summary=summary_text,
        summary_length=len(summary_text), method="TextRank/LexRank (Graph-Based)"
    )

@app.post("/summarize-text/abstractive", response_model=SummaryOutput, tags=["Summarization"])
def api_summarize_text_abstractive(payload: AbstractiveSummarizationInput):
    try:
        summary_text = generate_abstractive_summary(
            payload.text, max_length=payload.max_length, min_length=payload.min_length
        )
        return SummaryOutput(
            original_text_length=len(payload.text), summary=summary_text,
            summary_length=len(summary_text), method="Abstractive (Hugging Face BART)"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

@app.post("/tokenize-text", response_model=TokenizerOutput, tags=["Tokenization"])
def api_tokenize_text(payload: TextInput):
    tokens = tokenize_text(payload.text)
    return TokenizerOutput(original_text=payload.text, tokens=tokens, token_count=len(tokens))

@app.post("/analyze-sentiment", response_model=SentimentOutput, tags=["Sentiment Analysis"])
def analyze_review_sentiment(payload: TextInput):
    sentiment = predict_sentiment(payload.text)
    return SentimentOutput(original_text=payload.text, predicted_sentiment=sentiment)

@app.post("/extract-entities", response_model=NerOutput, tags=["Named Entity Recognition"])
def api_extract_entities(payload: TextInput):
    entities = extract_named_entities(payload.text)
    return NerOutput(original_text=payload.text, entities=entities)

@app.post("/visualize-entities", tags=["Named Entity Recognition"])
def api_visualize_entities(payload: TextInput):
    html_content = visualize_entities(payload.text)
    return Response(content=html_content, media_type="text/html")

@app.post("/process-words", response_model=WordProcessingOutput, tags=["Word Processing"])
def api_process_words(payload: WordListInput):
    processed_results = process_word_list(payload.words)
    return WordProcessingOutput(results=processed_results)

@app.post("/analyze-morphology", response_model=MorphologyOutput, tags=["Morphological Analysis"])
def api_analyze_morphology(payload: WordListInput):
    analysis_results = analyze_morphology_list(payload.words)
    return MorphologyOutput(results=analysis_results)
