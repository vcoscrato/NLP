# Imports
from numpy import logical_and
from pandas import DataFrame, concat, read_csv, read_table
from nltk import word_tokenize, pos_tag_sents
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, LeaveOneOut
from pickle import dump

# Dataset and sentilex
data = read_csv('Anotacoes3.csv', encoding='UTF-8')
sentilex = read_table('SentiLex-flex-PT01.txt', sep=';', header=None)

# Tagger
data['tag'] = pos_tag_sents(data['Sentença'].apply(word_tokenize).tolist(), lang='pt')
counts = []
for k in range(len(data)):
    counts.append(Counter([j for i, j in data['tag'][k]]))
dmm = DataFrame(counts).fillna(0)
tags = list(dmm)

# Bag-of-words
vec = CountVectorizer(min_df=0.01, max_df=0.5)
dtm = DataFrame(vec.fit_transform(data['Sentença'].values).toarray(), columns=vec.get_feature_names())
tokens = list(dtm)

# Sentiment
sentilex = sentilex.loc[sentilex.iloc[:, 3] != 'POL=0', :]
sentilex.drop([1, 2, 3, 4], axis=1, inplace=True)
sentilex[0].replace(to_replace='\..*$', value='', regex=True, inplace=True)
sent_terms = ','+sentilex.apply(','.join)[0]+','

sent_tokens = []
for i in tokens:
    if ','+i+',' in sent_terms:
        sent_tokens.append(i)

sent = dtm.loc[:, sent_tokens].apply(sum, axis=1)

# Model train and validation
x = concat([dtm, dmm, sent], axis=1)
y = data['Classe']

classifier = GradientBoostingClassifier()

cv = cross_val_predict(classifier, x, y, cv=LeaveOneOut())
acuracias = [sum(cv == y)/len(y),
             sum(logical_and(cv == 'F', y == 'F')) / sum(y == 'F'),
             sum(logical_and(cv == 'NF', y == 'NF')) / sum(y == 'NF')]

classifier.fit(x, y)

# Save files
with open('acuracias.pkl', 'wb') as f:
    dump(acuracias, f)

with open('classifier.pkl', 'wb') as f:
    dump(classifier, f)

with open('tokens.pkl', 'wb') as f:
    dump(tokens, f)

with open('tags.pkl', 'wb') as f:
    dump(tags, f)

with open('sent_tokens.pkl', 'wb') as f:
    dump(sent_tokens, f)
