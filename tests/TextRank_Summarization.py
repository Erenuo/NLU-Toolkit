# Gerekli modülleri sumy kütüphanesinden içe aktar
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer # TextRank'e çok benzer bir algoritma

news_article = """
Artificial intelligence has revolutionized numerous industries in recent years, transforming how businesses operate and people work.
Machine learning algorithms now power everything from recommendation systems on streaming platforms to autonomous vehicles navigating city streets.
Companies like Google, Microsoft, and OpenAI have invested billions of dollars in developing advanced AI systems that can understand and generate human language.
The rise of large language models has enabled chatbots to engage in sophisticated conversations, assist with creative writing, and even help with programming tasks.
However, concerns about AI safety, job displacement, and ethical implications continue to grow among researchers and policymakers.
Many experts argue that proper regulation and guidelines are essential to ensure AI development benefits humanity while minimizing potential risks.
The future of artificial intelligence remains uncertain, but its impact on society will undoubtedly continue to expand in the coming decades.
"""

# 1. Metni ayrıştırmak (parse) için gerekli araçları ayarla
parser = PlaintextParser.from_string(news_article, Tokenizer("english"))

# 2. Özetleyiciyi oluştur (LexRank)
summarizer = LexRankSummarizer()

# 3. Özetlenecek cümle sayısını belirle ve özetle
num_sentences_in_summary = 3
summary = summarizer(parser.document, num_sentences_in_summary)

# 4. Özet cümlelerini birleştir ve yazdır
final_summary = " ".join([str(sentence) for sentence in summary])
print("--- Summary (3 sentences) ---")
print(final_summary)
