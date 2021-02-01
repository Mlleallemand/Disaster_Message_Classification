import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine

import errno # to get error number for FileNotFound error
import os
from os import path

def load_data(messages_filepath, categories_filepath):
    """ Loads and merges csv-datasets from specified input paths on disk. 
        Input datasets are assumed to be csv files. This function uses pandas
        read_csv method and cannot be applied to SQL databases.
    
        Args:
        -----
            messages_filepath : (str)
                filepath to messages data
            categories_filepath : (str)
                filepath to message categories data
        
        Returns:
        --------
            pandas DataFrame Object
                merged data from the two specified input files. Data is
                assumed to contain a common ID column which is used to perform
                the merge on.
        
        Raises:
        -------
            FileNotFoundError
                if file is not on path or path is incorrect
    """
    
    # load messages data
    if not path.exists(messages_filepath):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), 
                                messages_filepath)
    messages = pd.read_csv(messages_filepath)
    
    # load categories data
    if not path.exists(categories_filepath):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), 
                                categories_filepath)
    categories = pd.read_csv(categories_filepath)
    
    return pd.merge(messages, categories, on='id')

def clean_data(df):
    """ Cleans an input DataFrame and returns the cleaned DataFrame: 
        - converts message categories into dummy-encoded columns named after
          message type
        - eliminates the original categories column
        - removes duplicate
        
        Args:
        -----
            df : DataFrame object containing messages and message categories from 
                 disaster Tweets
        
        Returns:
        --------
            cleaned DataFrame object as specified above
    """
    # Create a dataframe of the 36 individual category columns.
    # By applying .str.split twice, we automatically get a list of 
    # ['new column name', value] as entry for each cell. 
    categories = df['categories'].str.split(pat=';', expand=True).apply(
        lambda x: x.str.split('-'))
    
    # Extract column names from cells and store in array for later assignment.
    # Convert cell values to integer and delete column name from cell.
    new_columnames = []
    df_cat = categories.copy()
    for col in categories.columns:
        df_cat[col] = pd.to_numeric(categories[col].apply(lambda x: x[1]))
        new_columnames.append(categories.loc[0, col][0].lower().strip())
    
    # Assign new column names and join with original dataframe
    df_cat.columns = new_columnames;
    df.drop(columns='categories', inplace=True)
    df = pd.merge(left=df, right=df_cat, how='outer', left_index=True, 
                  right_index=True)
    
    # remove duplicates
    df.drop_duplicates(inplace=True)
    
    return df
        
def save_data(df, database_filename):
    """ Saves a dataframe to a SQL database defined by database_filename using
        pandas to_sql method.
    
        Args:
        -----
        df : DataFrame object that should be written to SQL database
        
        Returns: None            
    """
    if path.exists(database_filename):
        os.remove(database_filename)
        print('Database already existed. Database was removed and will be' + 
              ' overwritten \n')
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql(database_filename, engine, index=False)

def main():
    """ .
    
        Args:
        -----
        :
            
        
        Returns:
        --------
        :
            
    """ 
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

    print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
          .format(messages_filepath, categories_filepath))
    df = load_data(messages_filepath, categories_filepath)

    print('Cleaning data...')
    df = clean_data(df)

    print('Saving data...\n    DATABASE: {}'.format(database_filepath))
    save_data(df, database_filepath)

    print('Cleaned data saved to database!')
    
    #else:
    #    print('Please provide the filepaths of the messages and categories '\
    #          'datasets as the first and second argument respectively, as '\
    #          'well as the filepath of the database to save the cleaned data '\
    #          'to as the third argument. \n\nExample: python process_data.py '\
    #          'disaster_messages.csv disaster_categories.csv '\
    #          'DisasterResponse.db')


if __name__ == '__main__':
    main()