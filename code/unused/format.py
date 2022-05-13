

datafiles = ['yake1', 'yake3', 'yake5']

for datafile in datafiles:
    f = open(f'../outputs/gpt2/{datafile}_output.csv', 'r', encoding='utf-8')
    data = f.readlines()

    for line in data:
        d = line.split(',')
        print(len(d))