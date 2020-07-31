import numpy as np
import pandas as pd
import re
import spacy
import string

import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')

# from text_prep_config import UNICODE_EMO

import emoji

# !pip install emot
from emot.emo_unicode import UNICODE_EMO, EMOTICONS

import matplotlib.pyplot as plt
import seaborn as sns

import flair


## lower casing
def lower_casing(text):
    return str(text).lower()

## punctuation removal
PUNCT_TO_REMOVE = '!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~.' #string.punctuation
def remove_punctuation(text):
    """function to remove the punctuation"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

## stopword removal
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
## appending hinglish stopwords
hinglish = open("hinglish", encoding="utf-8")
hinglish_stopwords = hinglish.read().split('\n')

STOPWORDS = set(stopwords.words('english'))
STOPWORDS.update(hinglish_stopwords)

def remove_stopwords(text):
    """function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

## Stemming: Remove suffix and give the root of the word (Very fast, latency is importance)
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])


## Lemmatization: Remove suffix and check whether it is present in dictionary
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
def lemmatize_words(text):
    pos_tagged_text = nltk.pos_tag(str(text).split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

## removing emojis
def remove_emoji(string):
    
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    
    
    return emoji_pattern.sub(r'', string)

## url removal
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

## html removal
def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)

## handling contractions
from text_prep_config import CONTRACTIONS_DICT
contraction_words = set(CONTRACTIONS_DICT.keys())
def expand_contractions(text):
    new_text = []
    for w in str(text).split():
        if w in contraction_words:
            new_text.append(CONTRACTIONS_DICT[w])
        else:
            new_text.append(w)
    return " ".join(new_text)

## single letter words removal
def remove_single_letters(text):
    return " ".join([word for word in str(text).split() if len(word)>1])

##
def text_preprocessing(text):
    text = lower_casing(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text)
#     text = stem_words(text)
    text = lemmatize_words(text)
    text = remove_emoji(text)
    text = remove_urls(text)
    text = remove_html(text)
    text = expand_contractions(text)
    text = remove_single_letters(text)
    return text


## convert emojis
def convert_emojis(text):
    text = emoji.demojize(text)
    text = text.replace(":", " ")
    return text

## convert emoticons
def convert_emoticons(text):
    for emot in EMOTICONS:
        text = re.sub(u'('+emot+')', "_".join(EMOTICONS[emot].replace(",","").split()), text)
    return text

##
def sentiment_preprocessing(text):
    ## handling emoticons
    text = convert_emoticons(text)
    
    ## handling emojis
    text = convert_emojis(text)
    
    ## replacing "_" with " "
    text = text.replace("_", " ")
    return text

##
def visualise_sentiments(sentence):
    s = flair.data.Sentence(sentence)
    flair_sentiment.predict(s)
    total_sentiment = s.labels

    tokens = [token.text for token in s.tokens]
    ss = [flair.data.Sentence(s) for s in tokens]
    [flair_sentiment.predict(s) for s in ss]
    sentiments = [s.labels[0].score * (-1,1)[str(s.labels[0]).split()[0].startswith("POS")] for s in ss]

    data = {
          "Sentence":["SENTENCE"] + tokens,
          "Sentiment":[total_sentiment[0].score *(-1,1)[str(total_sentiment[0]).split()[0].startswith("POS")]] + sentiments,
    }

    plt.figure(figsize=(30, 1))
    sns.heatmap(pd.DataFrame(data).set_index("Sentence").T,center=0, annot=True, cmap = "PiYG", annot_kws={"size": 15})
    
    
##
flair_sentiment = flair.models.TextClassifier.load('en-sentiment')
def sentiment_(text):
    s = flair.data.Sentence(text)
    flair_sentiment.predict(s)
    total_sentiment = s.labels
    return total_sentiment[0]


