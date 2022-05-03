import nltk
from nltk.corpus import stopwords
from rake_nltk import Rake
from keybert import KeyBERT
import csv
from fastprogress.fastprogress import progress_bar
from transformers import BertForMaskedLM
import yake
from tqdm import tqdm
import json

stop_words = set(stopwords.words('english'))
kb = KeyBERT(model='all-mpnet-base-v2')

yake_1 = yake.KeywordExtractor(top=5, n=1, stopwords=stop_words)
yake_3 = yake.KeywordExtractor(top=5, n=3, stopwords=stop_words)
yake_5 = yake.KeywordExtractor(top=5, n=5, stopwords=stop_words)
yake1_data = {}
yake3_data = {}
yake5_data = {}

y1 = open('data/yake1.json', 'w', encoding='utf-8')
y3 = open('data/yake3.json', 'w', encoding='utf-8')
y5 = open('data/yake5.json', 'w', encoding='utf-8')


with open('../corpus_data/CNN_Articels_clean.csv', newline='', encoding='utf-8') as csvfile:
    reader = list(csv.DictReader(csvfile))
    for row in tqdm(reader):
        if row["Category"] == 'news':
            index = row["Index"]
            full_text = row['Article text']

            yake1_data[index] = [keyword[0] for keyword in yake_1.extract_keywords(full_text)]
            yake3_data[index] = [keyword[0] for keyword in yake_3.extract_keywords(full_text)]
            yake5_data[index] = [keyword[0] for keyword in yake_5.extract_keywords(full_text)]

json.dump(yake1_data, y1)
json.dump(yake3_data, y3)
json.dump(yake5_data, y5)

y1.close()
y3.close()
y5.close()