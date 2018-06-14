import numpy as np
import pandas as pd
from nltk import word_tokenize, pos_tag_sents
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, LeaveOneOut
import pickle

data = pd.read_csv('Anotacoes.csv', encoding='ISO-8859-1')
data['tag'] = pos_tag_sents(data['Sentença'].apply(word_tokenize).tolist(), lang='pt')
counts = []
for k in range(len(data)):
    counts.append(Counter([j for i, j in data['tag'][k]]))
dmm = pd.DataFrame(counts).fillna(0)

vec = CountVectorizer(min_df=0.01, max_df=0.5)
dtm = pd.DataFrame(vec.fit_transform(data['Sentença'].values).toarray(), columns=vec.get_feature_names())

x = pd.concat([dtm, dmm], axis=1)
y = data['Classe']

classifier = GradientBoostingClassifier()
print(np.mean(cross_val_score(classifier, dtm, y, cv=LeaveOneOut())))
print(np.mean(cross_val_score(classifier, dmm, y, cv=LeaveOneOut())))
print(np.mean(cross_val_score(classifier, x, y, cv=LeaveOneOut())))

classifier.fit(x, y)

with open('classifier.pkl', 'wb') as f:
    pickle.dump(classifier, f, 2)
