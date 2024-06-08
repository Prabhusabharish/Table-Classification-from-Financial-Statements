Overview
This project focuses on classifying data extracted from HTML tables in financial statements using machine learning algorithms. The goal is to predict the category of financial documents based on the content of the tables.

Project Structure
data: Directory containing HTML files of financial statements.
notebooks: Jupyter notebooks for data exploration, model training, and evaluation.
src: Source code files including data preprocessing, model training, and prediction functions.
load_data.py: Functions to load HTML tables and preprocess the data.
train_model.py: Functions to train machine learning models and perform hyperparameter tuning.
predict.py: Prediction function to classify HTML content into categories.
models: Saved trained models after hyperparameter tuning.
vectorizer.pkl: Pickle file for the text vectorizer used in the models.
Setup and Dependencies
Python 3.x
Install required packages using pip install -r requirements.txt.
Usage
Data Loading and Preprocessing:

Use load_data.py to load HTML tables and preprocess the data.
Model Training and Hyperparameter Tuning:

Run train_model.py to train machine learning models and perform hyperparameter tuning using grid search.
Prediction:

Use predict.py to predict the category of HTML content.
Example Usage:

See example_usage.ipynb in the notebooks directory for example code snippets.
Usage Examples
python train_model.py --data_dir /path/to/data --model_name RandomForest
python predict.py --html_file /path/to/html_file.html --model_name RandomForest


Prabakaran T
Email: prabhusabharish78@gmail.com
