# PROGRAMMER: JC Lopez  
# REVISED DATE: 06/11/2019
# PURPOSE: 
# Machine learning pipeline.
#   1. Load data from the SQLite database
#   2. Split the dataset into training and test sets
#   3. Build a text processing and machine learning pipeline
#   4. Train and tune a model using GridSearchCV
#   5. Output evaluation results on the test set
#   6. Export the final model as a pickle file
#
# BASIC USAGE:
#   $ python train_classifier.py <path to SQL database> <model filename>
#
# EXAMPLE:
#   $ python models/train_classifier.py data/DisasterResponse.db 
#         models/classifier.pkl


# Import Python libraries
import nltk
import numpy as np
import pandas as pd
import re
from sqlalchemy import create_engine
import sys
# Import NLTK classes and functions
from nltk.corpus import stopwords
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
# Import scikit-learn classes and functions
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier

# Download NLTK corpora
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    """Read records stored in SQLite database and return separate Pandas 
    objects for feature, targets, and category names.
    
    Args: 
        database_filepath (str)
    
    Returns: 
        X (Series): Features for the machile learning pipeline.
        Y (DataFrame): Labels for the machile learning pipeline.
        category_names (Index): Array with 36 category names. 
        
    """
    # Create database connectivity
    url = 'sqlite:///{}'.format(database_filepath)
    engine = create_engine(url)
    # Load DataFrame from database
    df = pd.read_sql("SELECT * FROM message", engine)
    
    # Define feature and target variables X and Y
    X = df['message']
    Y = df.loc[:,'related':'direct_report']
    # Get category names
    category_names = Y.columns
    
    return X, Y, category_names

def tokenize(text):
    """Clean, tokenize, and lemmatize text messages.
    
    Args: 
        text (str): Text content of a message.
    
    Returns: 
        lemmed (list): List of tokenized and lemmatized messages. 
        
    """
    # Make lowercase
    text = text.lower() 
    # Remove numeric characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 
    # Tokenize text 
    words = word_tokenize(text)
    # Remove stopwords
    words = [w for w in words if w not in stopwords.words("english")]
    # Lemmatize
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in words]
    
    return lemmed


def build_model():
    """Build scikit-learn pipeline for text processing and machine
    learning tasks.
    
    Args: 
        None
    
    Returns: 
        Pipeline (sklearn.pipeline)
        
    """    
    # Build model pipeline
    pipeline = Pipeline([
        # Text pipeline: vectorize word count and normalize 
        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
            ])),
        # Estimator
        ('clf', MultiOutputClassifier(
            RandomForestClassifier(random_state=42)
            ))
        ])
    # Pipeline parameters for grid search CV
    parameters = {
        #'text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        #'text_pipeline__vect__max_features': (None, 1000),
        'text_pipeline__tfidf__use_idf': (True, False),
        #'clf__estimator__min_samples_leaf': [1, 5],
        'clf__estimator__n_estimators': [10, 100]  
        }
    
    # Instantiate the GridSearchCV object
    model = GridSearchCV(pipeline, param_grid=parameters)
    
    return model


def best_params(model):
    """Print parameter setting that gave GridSearchCV best results on the hold out data.
    
    Args: 
        model (sklearn.pipeline): Fitted GridSearchCV pipeline.
        
    """
    print('  GridSearchCV best results:')
    max_len = max([len(key) for key in model.best_params_.keys()])
    string = '    {:<' + str(max_len + 1) + '}{:>7}'
    for key, value in model.best_params_.items():
        print(string.format(key + ':', value))
    
    return None            


def evaluate_model(model, X_test, Y_test, category_names):
    """Run test data through text processing and machine learning
    pipeline for prediction. And print results of scikit-learn's 
    classification_report(): f1-score, precision, and recall.
    
    Args: 
        model (sklearn.pipeline): Fitted scikit-learn pipeline.
        X_test (Series): Test features for prediction. 
        Y_test (DataFrame): Test labels for evaluating predictions.
        category_names (Index): Array with 36 category names. 

    Returns: 
        None
        
    """
    # Make target predictions using the fitted model
    Y_pred = model.predict(X_test)
    
    # For each target, print classification_report
    # f1-score, precision, and recall
    for index, column in enumerate(category_names):
        print('Category: **{}**'.format(column))
        print(classification_report(Y_test[column], Y_pred[:,index]))

    return None
    
   
def save_model(model, model_filepath):
    """Save fitted model in a pickle file.
    
    Args: 
        model (sklearn.pipeline): Fitted scikit-learn pipeline.
        model_filepath (str): Filename (pkl) for the the pickled model.
    
    Returns: 
        None
        
    """
    # Save best classifier out of GridSearchCV()
    joblib.dump(model.best_estimator_, model_filepath) 

    return None


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.99)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        best_params(model)
        
        print('Evaluating model...')
        #evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()