Used 2 different models: 
    1) LSTM Model, trained using a dataset of roughly 10k article titles
    2) GPT-2 finetuned model, using the same dataset of roughly 10k article titles
All articles used were under the "news" category. 

The test/development? corpus was size 1611 for the LSTM model, but only size 20 for the GPT-2 model, due to hardware and time-constraints.

The keywords of the testing corpus were extracted using the YAKE! library, because its light-weight, simple to use, and its benchmark tests showed it can outperform other state-of-the art methods.

If time permits, maybe another method of keyword extraction will be implemented, or a better set of stop words (beyond a generic list from nltk).

The top 5 keywords/phrases were extracted.

First limiting the size of the word/phrase to 1, so only a single word.
Then, limiting the size of the phrase to 3, so up to three words could be considered 'key'.
Then finally, limiting the size of the phrase to 5, so up to five words could be considered 'key'.

Afterwards, these keyword/phrases were inputted into both models.
Again, the LSTM model received all 5 keywords, of all 1611 titles (i.e. generated a total of 8055 titles)
The GPT-2 model only received 3 keywords, from 20 titles (due to time and hardware constraints, generating a total of 60 titles)

These titles were then compared to the original articles title, using cosine similarity (from sklearn), and the results are below:

LSTM Model:
yake1 : 0.06385564597491072
yake1 best cosine: 0.49630705945783515, index: 739, result: 0
Generated: Skin Whitening What Is What We Need To Know About The
Actual: Skin whitening: What is it, what are the risks and who profits?
yake3 : 0.06427950108759217
yake3 best cosine: 0.5448954693228305, index: 1942, result: 2
Generated: Sir John Major Russia Fast Facts In The Us And A Man Of
Actual: John Major Fast Facts
yake5 : 0.06427181437205604
yake5 best cosine: 0.49958147679900644, index: 813, result: 4
Generated: Leymah Gbowee The World And The Most Of 500 People Bike The
Actual: Leymah Gbowee: The people have awoken, we need to make the most of this moment

Best average result (6.5% similar)
Best result (54% similar)

GPT-2 Model:
yake1 : 0.045118583378910664
yake1 best cosine: 0.18817272215521147, index: 165, result: 1
Generated: 1) Cnn'It Is A Warzone,' Says Migrant 2) Russian Agreement Puts Ukraine In Eu 'Black Hole'
Actual: 'Help us, we're stranded': International students say they're trapped in northeast Ukraine
yake3 : 0.03172485258032359
yake3 best cosine: 0.31745314091816873, index: 73, result: 1
Generated: Russia Invades Ukraine To 'Hijack' Key Institutions
Actual: Kenya's UN ambassador slams Russia and compares Ukraine crisis to Africa's colonial past
yake5 : 0.06005158784842297
yake5 best cosine: 0.2630507988611436, index: 73, result: 2
Generated: Amazon Prime Video Fast Facts A Fast Facts For The Week
Actual: Amazon's going to Nollywood -- and its deals with studios could shake up one of the world's most prolific filmmaking hubs

Best average result (6.01% similar )
Best result (31% similar)



GPT-2 excelled with a larger limit on the keyword/phrase, nearly 50% increase of single keyword, and nearly double that of the 3 keyword.
LSTM had nearly the same results across the board, however surprisingly yake-3-keyword performed the best.

I would imagine that yake-5-keyword would stay the best for GPT-2, and would outperform yake-3-keyword for the LSTM Model, with some improvements to the keyword extraction, as well as our stop word list.

Possible imporvements:
 - Improve stopwords list
 - Different category titles (news is generally pretty broad)
 - Larger testing corpus for GPT-2 -- The inputs that generated the best results for LSTM weren't tested on GPT-2 due to time constraints
 