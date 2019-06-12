# PROGRAMMER: JC Lopez  
# REVISED DATE: 06/11/2019
# PURPOSE: 
# Data cleaning pipeline.
#   1. Load the messages and categories datasets
#   2. Merge the two datasets
#   3. Clean the data
#   4. Store it in a SQLite database
#
# BASIC USAGE:
#   $ python process_data.py <path to messages CSV file>
#         <path to categories CSV file> <database filename>
#
# EXAMPLE:
#   $ python data/process_data.py data/disaster_messages.csv 
#         data/disaster_categories.csv data/DisasterResponse.db


# Import Python libraries
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import sys

def load_data(messages_filepath, categories_filepath):
    """Read in the text CSV files — messages and categories — and return
    the merged dataset.
    
    Args: 
        messages_filepath (str): Path to CSV file containing messages.
        categories_filepath (str): Path to CSV file containing category 
            labels.
    
    Returns:
        df (DataFrame): Merged dataset of messages and category 
            labels.
        
    """
    # Read in messages dataset
    messages = pd.read_csv(messages_filepath, sep=',')
    # Read in categories dataset
    categories = pd.read_csv(categories_filepath, sep=',')
    
    # Merge datasets 
    df = pd.merge(messages, categories, how='left', on='id')
    return df


def clean_data(df):  
    """Clean the messages and categories dataset:

    1. Expand the string in the 'categories' column into multiple columns, 
        so that each column corresponds to one of the 36 labels.
    2. Remove the string values from all the category columns and 
        replace with appropriate numeric value.
    3. Drop the original 'categories' column.
    4. Drop duplicate rows.  
    
    Args: 
        df (DataFrame): Merged dataset of messages and category 
            labels.
    
    Returns: 
        df (DataFrame): Clean dataset with 'categories' columns 
            expanded into multiple columns with numeric values. 
        
    """
    # Create DF of the 36 individual category columns
    categories = df['categories'].str.split(";", expand=True)
    
    # Get category names and assign to columns 
    category_colnames = categories.iloc[0].apply(lambda string: string[0:-2])
    categories.columns = category_colnames
    
    # Convert values to binary
    for column in categories:
        # Set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1:]
        # Convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # Drop original categories column from dataframe
    df.drop(axis=1, columns='categories', inplace=True)
    
    # Concatenate dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis='columns')
    
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df

def save_data(df, database_filename):
    """Write records stored in the clean DataFrame into SQLite database.
    
    Args: 
        df (DataFrame): Output of clean_data(). Clean dataset with 
            'categories' columns expanded into 36 columns.
        database_filename (str)
    
    Returns: 
        None
        
    """
    # Build URL
    url = 'sqlite:///{}'.format(database_filename)
    
    # Create SQLite connection
    engine = create_engine(url)

    # Write dataset into SQLite database
    df.to_sql('message', engine, index=False)  

    return None


def main():
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
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()