# import libraries
import pdb
import sys
import pandas as pd
import numpy as np
import sqlite3
import re
import seaborn as sns
import os.path as path
import pickle
import datetime
import pdb
import joblib

import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# from pywsd.utils import lemmatize_sentence

from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
# from sklearn.pipeline import FeatureUnion


def load_data(database_name):
    """ Loads all data from a a SQL database for the specified input 
        filename and returns the data as a pandas DataFrame object.
    
        Args:
        -----
        database_name : str
            table name of the SQL database, e.g. "table_name" for SQL 
            database at sqlite:///table_name.db which contains a table
            named "table_name"
            
        Returns:
        --------
        X : numpy array holding message category information in binary format
        Y : numpy Series holding messages (converted to English) as strings
        category names : numpy array of column names for X        
    """
    engine = create_engine(('sqlite:///' + database_name))
    db_conn = sqlite3.connect(database_name)

    # if there might be more than one table, write loop based on:
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
    
    # related only holds value 1, id column is not necessary for classifier
    df.drop(columns=['related', 'id'], inplace=True)
    
    # convert to matrix
    X = df['message'].values.tolist()
    
    Y = df.select_dtypes(include=np.number);
    category_names = Y.columns.values;
    
    Y = np.array(Y);
    print(Y.shape)          
    return X, Y, category_names

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
    
    # Option 1 -------------------------------------------
    # uncomment to use pywsd to do lemmatization
    # pyswd uses part-of-speech tagging before lemmatizing
    # resulting in slightly better performance. Please
    # read note in Readme for necessary packages.
    # ----------------------------------------------------
    # clean_tokens = lemmatize_sentence(text)
    
    # Option 2 -------------------------------------------
    # uncomment following lines to use classical word net
    # lemmatizer without POS tagging
    # ----------------------------------------------------
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stopword_list = set(stopwords.words('english'))
    clean_tokens = [];
    for tok in tokens:
        # remove English stopwords before lemmatizing
        if tok not in stopword_list:
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)

    return clean_tokens               
                
def build_model():
    """ Builds and returns a Pipeline object consisting of a CountVectorizer,
        TF-IDF Transformer and a MultiOutputClassifier
    
    Args: None
    -----
    
    Returns: Pipeline object
    --------
    """

    model = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(
            KNeighborsClassifier(
            weights = 'distance',  # best params GS
            metric = 'minkowski')
            )
        )
    ])
    return model

def build_GS_model():
    """Builds and returns a Pipeline object consisting of a CountVectorizer,
       TF-IDF Transformer and a MultiOutputClassifier. The model is optimized
       using GridSearch.

    Args: None
    -----
    
    Returns: GridSearch object
    --------
    """
        # Default parameters to optimize for the model above:
   
    # 'vect__analyzer': 'word',
    # 'vect__binary': False,
    # 'vect__decode_error': 'strict',
    # 'vect__dtype': numpy.int64,
    # 'vect__encoding': 'utf-8',
    # 'vect__input': 'content',
    # 'vect__lowercase': True,
    # 'vect__max_df': 1.0,
    # 'vect__max_features': None,
    # 'vect__min_df': 1,
    # 'vect__ngram_range': (1, 1),
    # 'vect__preprocessor': None,
    # 'vect__stop_words': None,
    # 'vect__strip_accents': None,
    # 'vect__token_pattern': '(?u)\\b\\w\\w+\\b',
    # 'vect__tokenizer': <function __main__.tokenize(text)>,
    # 'vect__vocabulary': None,

    # 'tfidf__norm': 'l2',
    # 'tfidf__smooth_idf': True,
    # 'tfidf__sublinear_tf': False,
    # 'tfidf__use_idf': True,

    # 'clf__estimator__algorithm': 'auto',
    # 'clf__estimator__leaf_size': 30,
    # 'clf__estimator__metric': 'minkowski',
    # 'clf__estimator__metric_params': None,
    # 'clf__estimator__n_jobs': None,
    # 'clf__estimator__n_neighbors': 5,
    # 'clf__estimator__p': 2,
    # 'clf__estimator__weights': 'uniform',
    # 'clf__estimator': KNeighborsClassifier(),
    # 'clf__n_jobs': None

    parameters = {
        'text_pipeline__tfidf__smooth_idf': [False, True],
        'clf__estimator__weights': ['uniform', 'distance'],
        'clf__estimator__metric': ['minkowski','braycurtis']
    }

    pipeline = Pipeline([
        # Feature 1: text pipeline for preprocessing
        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ])),
        # Feature 2: Classifier to categorize texts
        ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    ])
    # create grid search object
    model =  GridSearchCV(pipeline, param_grid=parameters, verbose=True)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """ Evaluates the trained model and returns a dictionary with the
        results.
        
        Args:
        -----
        model : Pipeline object containing the fitted and trained machine 
                learning model.
        X_test : array-like test data with size 
                (#messages x #message categories)
        Y_test : array like containing correct message categories for X
        
        category_names : array
            labels for each of the message categories. The order of labels
            must - in ascending order - correspond to the number of the
            target categories in Y. 
        
            Example
            -------
            Y = [0 2 1] then category_names should be ['category 1',  
            'category 2', 'category 3'] for the following assignment:
            
            category 1 -> value 0 in Y
            category 2 -> value 1 in Y
            category 3 -> value 2 in Y

        Returns:
        --------
        report : str
            String containing the full report including precision, recall,
            f1-scores and support values for each category in Y (applied
            separately)
            
            Additionally, summed information is given as micro average, macro
            average, weighted avg and samples avg at the end of the table. 
            For details see section 3.3.2.9.2. in the scikit learn user guide
            on model evaluation on:
            
            https://scikitlearn.org/stable/modules/model_evaluation.html
    
    """
    Y_pred = model.predict(X_test)

    try:
        print("The best parameters for the model were found to be: \n")
        print(model.best_params_)
    except: AttributeError

    report = classification_report(
        y_true = Y_test, # ground truth (correct) target values.
        y_pred = Y_pred, # predicted target values
        target_names = category_names
    );
    print(report)
    return report, Y_pred

def report_to_dict(report):
    """ Converts a classification report for a MultiLevel Classifier as
        generated by sklearn's 'classification_report' function into a
        dictionary that can be used to plot the evaluation results as a
        heatmap with Seaborn.
        
        Args:
        -----
        report : str Classification Report as output by sklearn
        
        
        Returns:
        --------
        df : pandas DataFrame object containing report information.
             Please note that the summary information at the end of the 
             report is not included.
    """
    # split report by newline characters, strip leading & trailing whitespace
    single_line_str = [re.sub(r' +', ' ', x.lstrip().rstrip()) for x in 
                       report.split('\n')]
    
    # split each line by whitespace, resulting in [message, value, value,...]
    df = pd.DataFrame([x.split(' ') for x in single_line_str])

    # last column contains only NaN and the summaries, can be dropped
    df.drop(columns=df.columns[-1:], inplace=True)
    
    # assign message categories as column labels and set messages as index
    df.columns=['message category', 'precision', 'recall', 
                'f1 score', 'support']
    
    # clear former header (are now col labels) + newline row and cut off
    # average information at the end
    df = df[2:-6]
    df = df.set_index('message category',drop=True).astype(np.number)
    
    return df
    
def plot_report(report):
    """ Plots a report as generated by evaluate_model. If input is a report
        string, the report is converted to a dataframe first and the
        dataframe is returned.
        
        Args:
        -----
        report : str or DataFrame object
        
        Returns:
        --------
        report : as DataFrame object if input was string, else None
        
    """
    # if input is string, convert to report dictionary
    if isinstance(report, str):
        print('Converting input string to dict...')
        data = report_to_dict(report);
        print('... completed')
    else:
        data = report.copy();
    
    # w, h = plt.figaspect(2) # height should be 5x the width
    fig = plt.figure(figsize=(6.4, 11.6))

    gs = plt.GridSpec(1, 2, width_ratios=[2, 1], height_ratios=[1])

    # prepare data
    data.sort_values(by=['precision', 'recall'], inplace=True)

    # plot data for precision, recall and f1-score 
    # sorted by precision and recall
    fig.add_subplot(gs[0])
    sns.heatmap(data[data.columns.values[:-1]],
                annot=True, 
                fmt=".2f",
                yticklabels=True,
                )

    # prepare support data as nx2 array, as heatmap does not take 1D data
    data[' '] = data['support'].copy()

    fig.add_subplot(gs[1])
    # Second plot with only the last column
    sns.heatmap(data[['support', ' ']],
                yticklabels = False)

    plt.tight_layout()
    plt.savefig('model_classification_results.png')
    
    return data
    
def save_model(model, model_filepath, report_str, report_df, 
               Y_pred, X_test, Y_test):
    """ Saves the trained model 'model' to disk using the filepath specified.
        If parameters are given for report string and dataframe, these are
        saved as well.
    
        Args:
        -----
        model : Classifier or Pipeline object. Trained model to save to disk.
        model_filepath : path to save model under, default './model/'
        report_str : classification report as string
        report_df : classification report as DataFrame object
        Y_pred : array-like of predicted class labels
        X_test : message input data used for prediction
        Y_test : array-like of correct class labels used in model evaluation
    """
    with open(model_filepath, 'wb') as output_file: 
        joblib.dump([model, report_str, report_df, Y_pred, X_test, Y_test], output_file)
        
def load_model(model_filepath):
    """ Loads a saved model from filepath.
    
        Args:
        -----
        model : Classifier or Pipeline object. Trained model to save to disk.
        model_filepath : path to save model under, default './model/'

        Returns:
        --------
        model : Pipeline object containing the trained model
        report_str : classification output as single string
        report_df : DataFrame
            classification output as DataFrame object with columns
            precision, recall, f1-score and support and message
            categories as index. Summary information is lost.
        Y_pred : predicted message labels
        X_test : numpy array of test data
        Y_test : numpy array of message labels used for testing
    
    """
    with open(model_filepath, 'rb') as input_file: 
        model, report_str, report_df, Y_pred, X_test, \
            Y_test = joblib.load(input_file)
        
    return model, report_str, report_df, Y_pred, X_test, Y_test
    
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:];

    print('Loading data...\n    DATABASE: {}'.format(database_filepath))
    X, Y, category_names = load_data(database_filepath)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                                        test_size=0.2)

    print('Building model...')
    model = build_model()  # uncomment to build standard model
    # model = build_GS_model()  # uncomment to use gridsearch to build model

    print('Training model...')
    model.fit(X_train, Y_train)

    print('Evaluating model...')
    report_str, Y_pred = evaluate_model(model, X_test, Y_test, category_names)
    report_df = report_to_dict(report_str)

    print('Saving model...\n    MODEL: {}'.format(model_filepath))
    save_model(model, model_filepath, report_str, report_df, 
               Y_pred, X_test, Y_test)
    print('Trained model saved!')

    plot_report(report_df)

#    else:
#        print('Please provide the filepath of the disaster messages database '\
#              'as the first argument and the filepath of the pickle file to '\
#              'save the model to as the second argument. \n\nExample: python '\
#              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()