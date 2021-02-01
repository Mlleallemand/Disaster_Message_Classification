import json
import pandas as pd
import numpy as np
import re
import joblib
import sqlite3
import pdb
import os
import path


from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from nltk.corpus import stopwords
nltk.download('stopwords')

from flask import Flask
from flask import render_template, request, jsonify

import plotly
from plotly.graph_objs import Bar, Scatter

from sqlalchemy import create_engine

app = Flask(__name__)

def tokenize(text):
    """Tokenizes message input data by replacing urls with
       placeholders, lemmatizing and replacing whitespace.
    
    Args:
    -----
    text : string containing the message to be lemmatized 
    
    Returns:
    --------
    clean_tokens : list of strings with tokens
    """

    url_regex = r'http[s]?:\/\/(?:www)?(?P<url_main>([a-zA-Z]|[0-9])+).' + \
            '(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|' +\
            '(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    #first part of the url is kept as there might be information in it
    text = re.sub(url_regex, r'urlplaceholder_\1', text)

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stopword_list = set(stopwords.words('english'))

    clean_tokens = []
    for tok in tokens:
        # remove English stopwords before lemmatizing
        if tok not in stopword_list:
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)
    print(clean_tokens)
    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')

# if there might be more than one table, write loop based on:
db_conn = sqlite3.connect('../data/DisasterResponse.db')
cur = db_conn.cursor()
tables = cur.execute(
    "SELECT name FROM sqlite_master WHERE type='table';"
).fetchall()

df_list = []  # list for storing read-in dataframes
for table in tables:
    table_name = table[0]  # fetchall returns list of tuples
    df_list.append(pd.read_sql_query(
        'SELECT * from "%s"' % table_name,db_conn)
    )
cur.close()
db_conn.close
if len(df_list) > 1:
    print('There is more than one table in the database', 
          'using only first')
df = df_list[0]

# load model
model, report_str, report_df, Y_pred, X_test, Y_test \
    = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # data for first barplot (number of messages per genre)
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # data for 2nd barplot (number of messages per category)
    message_categories = df.columns[3:].tolist()
    category_counts = []
    for message in message_categories:
        category_counts.append(np.sum(df[message]))

    # data for 3rd plot: average word length per genre
    df['n_words'] = df.message.copy()
    df.n_words = df.n_words.apply(lambda x: int(len(re.findall(r'\w+',x))))

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x = genre_names,
                    y = genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x = message_categories,
                    y = category_counts
                )
            ],

            'layout': {
                'title': '# Messages per Category',
                'yaxis': {
                    'title': "Message Count"
                },
                'xaxis': {
                    'title': "Message Category"
                }
            }
        },
        {
            'data': [
                Scatter(
                    x = df.n_words,
                    y = df.genre,
                    mode = 'markers'
                )
            ],

            'layout': {
                'title': 'Word Count Distribution per Genre',
                'yaxis': {
                    'title': "Genre"
                },
                'xaxis': {
                    'title': "# words"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()