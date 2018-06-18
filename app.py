from flask import Flask, request, render_template
from pickle import load
from sklearn.feature_extraction.text import CountVectorizer
from pandas import DataFrame, concat
from nltk import word_tokenize, pos_tag_sents
from collections import Counter

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

    with open('classes.pkl', 'rb') as f:
        classes = load(f)

    text = str(result['texto'])

    new_data = DataFrame(text.split('.'), columns=['Sentença'])
    new_data['tag'] = pos_tag_sents(new_data['Sentença'].apply(word_tokenize).tolist(), lang='pt')
    counts = []
    for k in range(len(new_data)):
        counts.append(Counter([j for i, j in new_data['tag'][k]]))
    dmm = DataFrame(counts).fillna(0)
    for i in range(len(classes)):
        if classes[i] not in dmm:
            dmm[classes[i]] = 0

    vec = CountVectorizer(vocabulary=tokens)
    dtm = DataFrame(vec.fit_transform(text.split('.')).toarray(), columns=vec.get_feature_names())

    with open('classifier.pkl', 'rb') as f:
        classifier = load(f)

    prediction = classifier.predict(concat([dtm, dmm], axis=1))
    proportion = 100*sum(prediction == 'F')/len(prediction)

    return render_template('result.html', prediction=proportion)

if __name__ == '__main__':
    app.run()