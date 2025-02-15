from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from transformers import pipeline
import streamlit as st
import re

st.header("TEXT SENTIMENT ANALYSIS")

input = st.text_input(placeholder="Enter or Paste the text here ...." , label="Text")

input = re.sub('[!*#.^&()~`"",+=?%$@:;<>{}|-]' ," ", string=input)

st.write(input)

tokens = word_tokenize(input)

st.write(tokens)

processed_words = []
lemmatizer = WordNetLemmatizer()
for word in tokens:
    if word in stopwords.words("english"):
        continue
    lemma = lemmatizer.lemmatize(word)
    processed_words.append(lemma)

st.write(processed_words)

model = pipeline("sentiment-analysis")
text = ' '.join(map(str, processed_words))
out = model.predict(text)
if out[0]["label"]=="POSITIVE":
    st.success("It is positive")
else:
    st.error("It is negative")

st.write(out)
