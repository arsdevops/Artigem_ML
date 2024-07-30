import pandas as pd
import joblib
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
import logging
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
import re
import json
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from collections import defaultdict
from urllib.parse import unquote

logger = logging.getLogger(__name__)

# Load model and vectorizer
def load_model_and_vectorizer_nb():
    try:
        clf = joblib.load('naive_bayes_model_new1.pkl')
        vectorizer1 = joblib.load('tfidf_vectorizer_new1.pkl')
        logger.info("Models and data loaded successfully.")
        return clf, vectorizer1
    except Exception as e:
        logger.error(f"Error loading models or data: {e}")
        raise

clf, vectorizer1 = load_model_and_vectorizer_nb()
lemmatizer = WordNetLemmatizer()

# Text cleaning and vectorization function
def clean_text(text):
    text = text.lower()
    text = text.replace('-', ' ')
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    text = ''.join([char for char in text if not char.isdigit()])
    text = ' '.join(text.split())
    return text

def clean_and_vectorize_nb(text):
    try:
        text = unquote(text)  # Decode URL-encoded text
        text = clean_text(text)
        vec_text = vectorizer1.transform([text])
        logger.info("Text cleaned and vectorized.")
        return vec_text
    except Exception as e:
        logger.error(f"Error in clean_and_vectorize: {e}")
        raise

def classify_input_nb(input_text):
    try:
        vec_text = clean_and_vectorize_nb(input_text)
        predicted_result = clf.predict(vec_text)[0]
        logger.info(f"Classification completed successfully for text: {input_text}")
        probabilities = clf.predict_proba(vec_text)[0]
        class_labels = clf.classes_
        logger.info("Input processed for prediction.")
        return predicted_result, probabilities, class_labels
    except Exception as e:
        logger.error(f"Error in classify_input: {e}")
        raise

# Dictionary to track retrain attempts
retrain_attempts = defaultdict(int)

def retrain_model_nb(new_data, new_category, new_subcategory):
    global retrain_attempts
    try:
        # Check for NaN values in input data
        if pd.isna(new_data) or pd.isna(new_category) or pd.isna(new_subcategory):
            logger.error("Input data contains NaN values.")
            raise ValueError("Input data contains NaN values.")
        
        key = f"{new_data}-{new_category}-{new_subcategory}"
        retrain_attempts.setdefault(key, 0)  # Initialize the counter if not already present
        retrain_attempts[key] += 1
        
        # Add the new data to the Excel file every time
        df = pd.read_excel('TrainingData-NB-Balanced-New.xlsx')
        new_entry = pd.DataFrame({
            'Description': [clean_text(unquote(new_data))],
            'Category': [new_category],
            'SubCategory': [new_subcategory]
        })
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_excel('TrainingData-NB-Balanced-New.xlsx', index=False)
        
        if retrain_attempts[key] >= 3:
            X_train = df['Description']
            y_train = df['Category'].str.cat(df['SubCategory'], sep='-')  # Combine categories for training only
            vectorizer1.fit(X_train)
            X_train_vec = vectorizer1.transform(X_train)
            clf.fit(X_train_vec, y_train)
            
            joblib.dump(clf, 'naive_bayes_model_new1.pkl')
            joblib.dump(vectorizer1, 'tfidf_vectorizer_new1.pkl')
            
            # Reset the counter after retraining
            retrain_attempts[key] = 0
            
            logger.info("Model retrained and saved.")
            return "Model retrained with the new dataset."
        
        logger.info(f"Retrain attempt {retrain_attempts[key]} for key: {key}")
        return "Not enough retrain attempts yet."
    except Exception as e:
        logger.error(f"Error in retrain_model: {e}")
        raise

def classify_items_nb(items):
    results = []
    errors = []
    for item in items:
        item_number = item.get('itemNumber')
        item_description = item.get('itemDescription', '').strip()
        if not item_description:
            errors.append(f"Item {item_number} is missing a description.")
            results.append({
                'itemNumber': item_number,
                'itemDescription': item_description,
                'category': 'Description Missing',
                'subcategory': None
            })
            continue
        try:
            result, _, _ = classify_input_nb(item_description)
            if '-' in result:
                category, subcategory = result.split('-', 1)
            else:
                category = result
                subcategory = None

            classified_item = {
                'itemNumber': item_number,
                'itemDescription': item_description,
                'category': category.strip(),
                'subcategory': subcategory.strip() if subcategory else None
            }
            results.append(classified_item)
        except Exception as e:
            errors.append(f"Error processing item {item_number}: {str(e)}")
    return results, errors

