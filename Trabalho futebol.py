from numpy import logical_and
from pandas import DataFrame, concat, read_csv
from nltk import word_tokenize, pos_tag_sents
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, LeaveOneOut
from pickle import dump

data = read_csv('Anotacoes2.csv', encoding='ISO-8859-1')
data['tag'] = pos_tag_sents(data['Sentença'].apply(word_tokenize).tolist(), lang='pt')
counts = []
for k in range(len(data)):
    counts.append(Counter([j for i, j in data['tag'][k]]))
dmm = DataFrame(counts).fillna(0)

vec = CountVectorizer(min_df=0.01, max_df=0.5)
dtm = DataFrame(vec.fit_transform(data['Sentença'].values).toarray(), columns=vec.get_feature_names())

x = concat([dtm, dmm], axis=1)
y = data['Classe']

classifier = GradientBoostingClassifier()

cv = cross_val_predict(classifier, x, y, cv=LeaveOneOut())
acuracias = [sum(cv == y)/len(y),
             sum(logical_and(cv == 'F', y == 'F')) / sum(y == 'F'),
             sum(logical_and(cv == 'NF', y == 'NF')) / sum(y == 'NF')]
with open('acuracias.pkl', 'wb') as f:
    dump(acuracias, f)

classifier.fit(x, y)

with open('classifier.pkl', 'wb') as f:
    dump(classifier, f)

with open('tokens.pkl', 'wb') as f:
    dump(list(dtm), f)

with open('classes.pkl', 'wb') as f:
    dump(list(dmm), f)
