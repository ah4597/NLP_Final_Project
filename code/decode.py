import json
import ijson


datafiles = ['yake1', 'yake3', 'yake5']
translation = {
    'yake1': 'Keyword of length 1',
    'yake3': 'Keyphrase of length 3',
    'yake5': 'Keyphrase of length 5'
}
decode = {
    'yake1_lstm': [
        (0.49630705945783515, 739, 0),
        (0.3868951156441445, 210, 1),
        (0.37225238290332247, 2139, 3),
        (0.3702568650509892, 1966, 4),
        (0.37011287544582205, 965, 0)
    ],
    'yake3_lstm': [
        (0.5448954693228305, 1942, 2),
        (0.5303652344294485, 1750, 2),
        (0.5052568105313082, 813, 2),
        (0.4840490271595034, 2396, 3),
        (0.44779752785327154, 2023, 1),
    ],
    'yake5_lstm': [
        (0.49958147679900644, 813, 4),
        (0.4688757007858698, 2023, 2),
        (0.4462427393437521, 938, 1),
        (0.42484324949588537, 739, 2),
        (0.39765429372344135, 1039, 2)
    ],

    'yake1_gpt2': [
        (0.19720980713320327, 174, 0),
        (0.18234940215907725, 3, 0),
        (0.17466696474795906, 165, 1),
        (0.14404089321633456, 170, 0),
        (0.13491027104118425, 165, 2)
    ],
    'yake3_gpt2': [
        (0.2106062151683532, 171, 0),
        (0.1818432159745451, 161, 2),
        (0.13505421969275536, 156, 2),
        (0.1269330902851618, 73, 1),
        (0.1238831642572425, 159, 0)
    ],
    'yake5_gpt2': [
        (0.2106062151683532, 171, 0),
        (0.1818432159745451, 161, 2),
        (0.13505421969275536, 156, 2),
        (0.1269330902851618, 73, 1),
        (0.1238831642572425, 159, 0)
    ]
}
for folder in ['lstm', 'gpt2']:
    output = open(f'../outputs/results/top_5_{folder}.txt', 'w', encoding='utf-8')
    original_titles = json.load(open('../corpus_data/test_titles.json', encoding='utf-8'))
    for datafile in datafiles:
        output.write(f'{translation[datafile]}\n')
        parser = ijson.parse(open(f'../outputs/{folder}/{datafile}_output.json'))
        data = {}  
        index = 0
        for prefix, event, value in parser:        
            if event == 'start_array':
                index = prefix
                data[index] = []
            if event == 'string':
                data[index].append(value)
            if event == 'end_array':
                pass
        
        for d in decode[f'{datafile}_{folder}']:
            output.write(f'Original Title: {original_titles[f"{d[1]}"]}\nBest Results: {data[f"{d[1]}"][d[2]]}\nCosine: {d[0]}\n\n')
        output.write('\n\n')