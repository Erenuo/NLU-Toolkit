from transformers import pipeline

# 1. Initialize the summarization pipeline
# This will download the model and tokenizer on the first run
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# 2. Define the text you want to summarize
long_text = """
Jupiter is the fifth planet from the Sun and the largest in the Solar System. 
It is a gas giant with a mass more than two and a half times that of all the other planets 
in the Solar System combined, but slightly less than one-thousandth the mass of the Sun. 
Jupiter is the third brightest natural object in the Earth's night sky after the Moon and Venus. 
It has been known to astronomers since antiquity. It is named after the Roman god Jupiter. 
When viewed from Earth, Jupiter can be bright enough for its reflected light to cast visible shadows, 
and is on average the third-brightest natural object in the night sky after the Moon and Venus.
"""

# 3. Generate the summary
# You can adjust min_length and max_length for the desired summary size
summary = summarizer(long_text, max_length=60, min_length=25, do_sample=False)

# 4. Print the result
print(summary[0]['summary_text'])