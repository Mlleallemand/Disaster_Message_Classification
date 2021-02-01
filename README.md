# Disaster Response Pipeline Project

This repository contains an analysis pipeline using MultiOutput Classification to categorize a set of Twitter messages into 36 different message categories. Twitter messages were recorded during natural disasters and message categorisation should aid help staff to allocate resources more efficiently.

Data and parts of the script were provided as part of the "Data Scientist" nanodegree from Udacity.com

## Necessary Python libraries

#### contained in standard library
* [datetime]
* [errno]
* [os]
* [sys]
* [re]

#### not contained in standard library
* [Numpy] version 1.18.5 
* [Pandas] version 1.0.5
* [Seaborn] version 0.11.1
* [matplotlib]
* [sqlite3]
* [sqlalchemy]
* [sklearn]
* [joblib]
* [pickle]
* [nltk] 
* [pywsd] version 1.2.4

## Repository contents

| Folder | Filename | Description|
| -------- | -------- | -----------|
| main | model_classification_results.png | classification results of the model, generated after training in train_classifier.py|
| data | process_data.py | script to load, preprocess and store 						   data in SQL database |
| data | disaster_categories.csv | categories for tweets |
| data | disaster_messages.csv | raw input data containing tweets |
| data | DisasterResponse.db | SQL database with cleaned data |
| models | train_classifier.py | Runs ML pipeline to train classifier and saves the trained model in the folder |
| models | classifier.pkl | trained model saved by train_classifier.py |
| app | run.py | script to run the Flask webapp |
| app/templates | master.html | necessary for webapp, provided bz Udacity |
| app/templates | go.html | necessary for webapp, provided by Udacity |

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
      If you want to use GridSearch to optimize model parameters,
      uncomment line 417 in train_classifier and comment line 416

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. In your web browser, go to http://0.0.0.0:3001/

### Summary:
Classification results are poor, even when using the Grid Search to optimize parameters. This can be seen by the very low recall values and also by trying out various messages in the WebApp.

Potential reasons / optimization options:
* using context by part-of-speech tagging
* training by different genres - as news contains often very many different topics in one message thus making classification more difficult.

### Note:
When using pywsd for better lemmatization performance, you need to install pyswd and then downgrade the wn package using

		pip install pyswd*
		pip install -U wn==0.0.23*

You also need to download the averaged perceptron tagger by adding the line below to the import statements in train_classifier.py

		nltk.download('averaged_perceptron_tagger')*


[Numpy]:<https://numpy.org>
[Scipy]:<https://scipy.org>
[Pandas]:<https://pandas.pydata.org/>
[Seaborn]:<https://seaborn.pydata.org/>
[scikit learn]:<https://scikit-learn.org/stable/>
[matplotlib]:<https://matplotlib.org/>
[sqlite3]:<https://www.sqlite.org/index.html>
[sqlalchemy]:<https://www.sqlalchemy.org/>
[sklearn]:<https://scikit-learn.org/stable/>
[joblib]:<https://joblib.readthedocs.io/en/latest/>
[pickle]:<https://docs.python.org/3/library/pickle.html>
[nltk]:<https://www.nltk.org/>
[pywsd]:<https://pypi.org/project/pywsd/>

[errno]:<>
[os]:<>
[sys]:<>
[re]:<>
[datetime]:<https://docs.python.org/3/library/datetime.html>