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
from numba import jit, cuda


with open('data/CNN_Articels_clean.csv', newline='', encoding='utf-8') as csvfile:
    reader = list(csv.DictReader(csvfile))
    data = {}
    for row in tqdm(reader):
        if row["Category"] == 'news':
            data[row["Index"]] = row["Second headline"]
    
    with open('data/test_titles.json', 'w', encoding='utf-8') as output:
        json.dump(data, output)