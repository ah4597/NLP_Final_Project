## OLD __ OUT OF USE

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import ijson


tfidf_vectorizer = TfidfVectorizer(analyzer="word")
datafiles = ['yake1', 'yake3', 'yake5']

output_file = open('../outputs/results/results_gpt2.txt', 'w', encoding='utf-8')

for datafile in datafiles:
    total = 0
    i = 0
    best_index = 0
    best_cosine = 0
    best_result = i
    all_cosines = []
    with open('../corpus_data/test_titles.json') as json_file:
        key = json.load(json_file)
        k = dict(list(key.items())[:20])
        with open(f'../outputs/gpt2/{datafile}_output.csv', 'r', encoding='utf-8') as f:
            data = f.readlines()
            for line in data:
                d = line.strip().split(',')
                index = d[0]

                string_key = k[index]
                string_list = d[1:len(d)-1]
                sparse_matrix = tfidf_vectorizer.fit_transform([string_key]+string_list)
                cosine = cosine_similarity(sparse_matrix[0,:],sparse_matrix[1:,:])
                i += len(string_list)
                for result in cosine:
                    for j in range(len(result)):
                        nested = result[j]
                        if best_cosine < nested:
                            best_cosine = nested
                            best_index = index
                            best_result = j
                        total += nested
                        all_cosines.append((nested, index, j))
                        
    average_cosine = total/i
    top_5_cosines = sorted(all_cosines, key=lambda item: item[0], reverse=True)[:5]
    for cosine, index, j in top_5_cosines:
        output_file.write(f'({cosine}, {index}, {j})\n')
    print(f'Average cosine for {datafile}: {average_cosine}')
    output_file.write(f'{datafile} : {average_cosine}\n')
    output_file.write(f'{datafile} best cosine: {best_cosine}, index: {best_index}, result: {best_result}\n')
