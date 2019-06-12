# Analysis of text messages for Disaster Response 

## ETL pipeline | NLP Machine Learning pipeline | Flask Web App

<br />

> Example of message classification screen in the web app
> message: "Help! My house is getting flooded by the rain."

<p align="center">
  <img src="docs/example-message.png" width="512" alt="screen-shot" />
</p>

---

This project is an exercise in natural language processing and machine learning pipelines, using a dataset which contains real messages that were sent during disaster events. The objective is to categorize the underlying events so that the messages can be redirected to an appropriate disaster relief agency.

The user interface is a web app where an emergency worker can input a new message and get classification results in several categories. The app also displays visualizations of the training data. 

This was originally built for a [Udacity](https://www.udacity.com/) program: [Data Science degree, Project 5](https://github.com/udacity/DSND_Term2). The data was provided by Udacity's partners at [Figure Eight](https://www.figure-eight.com/).


## Main files in the repository


**`data` folder:**

- `process_data.py`: Data cleaning pipeline. (1) Load CSV datasets; (2) merge and clean the data; (3) store the clean dataset in an SQLite database. 

- `disaster_messages.csv`: Dataset of text messages.

- `disaster_categories.csv`: Dataset of labels for 36 categories.

- `DisasterResponse.db`: SQLite database. Output of `process_data.py`.


**`models` folder:**

- `train_classifier.py`: Text processing and machine learning pipeline. (1) Load SQLite database; (2) split the dataset into training and test sets; (3) build a text processing and machine learning pipeline; (4) train and tune a model using GridSearchCV; (5) output evaluation results on the test set; (6) export the final model as a pickle file.

- `classifier.pkl`: Fitted model as a pickle file.


**`app` folder:**

- `run.py`: Python script for Flask app.

- `templates`: Directory with HTML files for Flask app.


**Jupyter Notebooks:**

- `ETL Pipeline Preparation.ipynb`: Jupyter notebook

- `ML Pipeline Preparation.ipynb`: Jupyter notebook

    
## Data Science motivation

Build text processing and machine learning pipelines using tools in [NLTK](https://www.nltk.org/) and [`scikit-learn`](https://scikit-learn.org/).


## Use `process_data.py`

```
python process_data.py <path to messages CSV file> 
	<path to categories CSV file> <database filename>
```

Example:
```
python data/process_data.py data/disaster_messages.csv 
	data/disaster_categories.csv data/DisasterResponse.db
```

## Use `train_classifier.py`

```
python train_classifier.py <path to SQL database> <model filename>
```

**Example:**
```
python models/train_classifier.py data/DisasterResponse.db  
	models/classifier.pkl
```


## Use `run.py`

- Run the Flask app from the `app` directory: `python run.py`.

- Open a web browser and go to http://0.0.0.0:3001/.


## Use example of Jupyter Notebooks for new users

The Jupyter Project highly recommends new users to install [Anaconda](https://www.anaconda.com/distribution/); since it conveniently installs Python, the Jupyter Notebook, and other commonly used packages for scientific computing and data science.

Use the following installation steps:

1. Download Anaconda.

2. Install the version of Anaconda which you downloaded, following the instructions on the download page.

3. To run the notebook:

```
jupyter notebook "ETL Pipeline Preparation.ipynb"
```

## Python version

3.7.1 (default, Oct 23 2018, 14:07:42) 


## Python libraries

The Jupyter Notebook and the Python modules require the following Python libraries:

- flask
- json
- matplotlib
- nltk
- numpy
- pandas
- plotly
- re
- sklearn
- sqlalchemy
- sys


## Acknowledgments

- [Figure Eight](https://www.figure-eight.com/) for the dataset containing real messages that were sent during disaster events.

- [Udacity: Data Scientist Nanodegree program](https://www.udacity.com/course/data-scientist-nanodegree--nd025) for instructions and web-app template.

- Jupyter Documentation: [Installing Jupyter Notebook](https://jupyter.readthedocs.io/en/latest/install.html).


## Author

Juan Carlos Lopez

- jc.lopezh@gmail.com
- [GitHub](https://github.com/jclh/)
- [LinkedIn](https://www.linkedin.com/in/jclopezh/)


## Contributing

1. Fork it (https://github.com/jclh/NLP-disaster-response/fork)
2. Create your feature branch (git checkout -b feature/fooBar)
3. Commit your changes (git commit -am 'Add some fooBar')
4. Push to the branch (git push origin feature/fooBar)
5. Create a new Pull Request




























