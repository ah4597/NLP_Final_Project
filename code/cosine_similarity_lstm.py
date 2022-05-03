from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import ijson


tfidf_vectorizer = TfidfVectorizer(analyzer="word")
datafiles = ['yake1', 'yake3', 'yake5']

output_file = open('../outputs/results/results_lstm.txt', 'w', encoding='utf-8')

for datafile in datafiles:
    total = 0
    i = 0
    best_index = 0
    best_cosine = 0
    best_result = i
    with open('../corpus_data/test_titles.json') as json_file2:
        key = json.load(json_file2)
        parser = ijson.parse(open(f'../outputs/lstm/{datafile}_output.json'))
        
        index = 0
        data = []
        try:
            for prefix, event, value in parser:
                if event == 'start_array':
                    index = prefix
                if event == 'string':
                    data.append(value)
                if event == 'end_array':
                    string_key = key[index]
                    string_list = data
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
                    index = 0
                    data = []
        except:
            print(index)
            print(data)
    
    average_cosine = total/i
    print(f'Average cosine for {datafile}: {average_cosine}')
    output_file.write(f'{datafile} : {average_cosine}\n')
    output_file.write(f'{datafile} best cosine: {best_cosine}, index: {best_index}, result: {best_result}\n')