from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import ijson


tfidf_vectorizer = TfidfVectorizer(analyzer="word")
datafiles = ['yake1', 'yake3', 'yake5']

output_file = open('results.txt', 'w', encoding='utf-8')

for datafile in datafiles:
    total = 0
    i = 0

    with open('data/test_titles.json') as json_file2:
        key = json.load(json_file2)
        parser = ijson.parse(open(f'data/{datafile}_output.json'))
        
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
                    print(string_key)
                    print(data)
                    sparse_matrix = tfidf_vectorizer.fit_transform([string_key]+string_list)
                    cosine = cosine_similarity(sparse_matrix[0,:],sparse_matrix[1:,:])
                    i += len(string_list)
                    for result in cosine:
                        for nested in result:
                            total += nested
                    index = 0
                    data = []
        except:
            print(index)
            print(data)


    """ with open(f'data/{datafile}_output.json', 'r') as json_file:

        data = json.load(json_file)
            with open('data/test_titles.json') as json_file2:
                key = json.load(json_file2)
                for index in tqdm(data):
                    string_key = key[index]
                    string_list = data

                    sparse_matrix = tfidf_vectorizer.fit_transform([string_key]+string_list)
                    cosine = cosine_similarity(sparse_matrix[0,:],sparse_matrix[1:,:])
                    i += len(string_list)
                    for result in cosine:
                        for nested in result:
                            total += nested """
    
    average_cosine = total/i
    print(f'Average cosine for {datafile}: {average_cosine}')
    output_file.write(f'{datafile} : {average_cosine}\n')