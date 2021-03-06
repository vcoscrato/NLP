from flask import Flask, request, render_template
from pickle import load
from sklearn.feature_extraction.text import CountVectorizer
from pandas import DataFrame, concat
from nltk import word_tokenize, pos_tag_sents
from collections import Counter
import re

app = Flask(__name__)


@app.route('/')
def home():

    with open('acuracias.pkl', 'rb') as f:
        ac = load(f)

    return render_template('home.html', total=ac[0], f=ac[1], nf=ac[2])


@app.route('/predict', methods=['POST'])
def predict():

    result = request.form

    with open('tokens.pkl', 'rb') as f:
        tokens = load(f)

    with open('tags.pkl', 'rb') as f:
        tags = load(f)

    with open('sent_tokens.pkl', 'rb') as f:
        sent_tokens = load(f)

    text = str(result['texto'])
    text = re.sub(r'\.+', ".", text).split('.')
    text = [re.sub(r'[^\w\s]', '', x).strip() for x in text]
    text = [x.strip() for x in text if x.strip()]

    new_data = DataFrame(text, columns=['Sentença'])
    new_data['tag'] = pos_tag_sents(new_data['Sentença'].apply(word_tokenize).tolist(), lang='pt')
    counts = []
    for k in range(len(new_data)):
        counts.append(Counter([j for i, j in new_data['tag'][k]]))
    dmm = DataFrame(counts).fillna(0)
    for i in range(len(tags)):
        if tags[i] not in dmm:
            dmm[tags[i]] = 0

    vec = CountVectorizer(vocabulary=tokens)
    dtm = DataFrame(vec.fit_transform(text).toarray(), columns=vec.get_feature_names())

    sent = dtm.loc[:, sent_tokens].apply(sum, axis=1)

    with open('classifier.pkl', 'rb') as f:
        classifier = load(f)

    prediction = classifier.predict(concat([dtm, dmm, sent], axis=1))
    proportion = 100*sum(prediction == 'F')/len(prediction)

    new_data['classe'] = prediction

    return render_template('result.html', prediction=proportion, table=new_data.to_html())


if __name__ == '__main__':
    app.run()