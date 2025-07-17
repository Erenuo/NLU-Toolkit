from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer

# LexRank, TextRank'e çok benzer bir graf tabanlı özetleme algoritmasıdır.
# Bu modül için başlangıçta yüklenecek ayrı bir model yoktur.
# Gerekli nesneler her çağrıda oluşturulur.

def generate_textrank_summary(text: str, num_sentences: int = 3) -> str:
    """
    Args:
        text (str): The input text to be summarized.
        num_sentences (int): The desired number of sentences in the summary.

    Returns:
        str: The generated summary as a single string.
    """
    # 1. Create a parser for the input text.
    # The tokenizer splits the text into sentences and words.
    parser = PlaintextParser.from_string(text, Tokenizer("english"))

    # 2. Initialize the summarizer.
    summarizer = LexRankSummarizer()

    # 3. Generate the summary with the specified number of sentences.
    summary_sentences = summarizer(parser.document, num_sentences)

    # 4. Join the summary sentences back into a single string and return.
    return " ".join([str(sentence) for sentence in summary_sentences])
