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

logger = logging.getLogger(__name__)

def load_models_and_vectorizer():
    try:
        category_model = joblib.load('logistic_regression_category_model.pkl')
        subcategory_model = joblib.load('logistic_regression_subcategory_model.pkl')
        vectorizer = joblib.load('mapping_tfidf_vectorizer.pkl')
        logger.info("Models and vectorizer loaded successfully.")
        return category_model, subcategory_model, vectorizer
    except Exception as e:
        logger.error(f"Error loading models or vectorizer: {e}")
        raise

category_model, subcategory_model, vectorizer = load_models_and_vectorizer()
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Load category-subcategory mapping from JSON
with open('mapping.json') as f:
    cat_subcat_map = json.load(f)

def preprocess(text):
    text = text.lower().strip()
    text = re.sub('<.*?>|[%s]|\d|\[[0-9]*\]' % re.escape(string.punctuation), ' ', text)
    text = re.sub('\s+', ' ', text).strip()
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    text = stem_text(text)
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    custom_stop_words = set(['new', 'pack', 'packs'])  # Add custom stopwords as needed
    all_stop_words = stop_words.union(custom_stop_words)
    words = word_tokenize(text)
    return ' '.join([word for word in words if word not in all_stop_words])

def lemmatize_text(text):
    words = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words]
    return ' '.join(lemmatized_words)

def stem_text(text):
    words = word_tokenize(text)
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def clean_and_vectorize(text):
    try:
        preprocessed_text = preprocess(text)
        vec_text = vectorizer.transform([preprocessed_text])
        logger.info("Text cleaned and vectorized: %s", preprocessed_text)
        return vec_text
    except Exception as e:
        logger.error(f"Error in clean_and_vectorize: {e}")
        raise

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
        text = clean_text(text)
        vec_text = vectorizer1.transform([text])
        logger.info("Text cleaned and vectorized.")
        return vec_text
    except Exception as e:
        logger.error(f"Error in clean_and_vectorize: {e}")
        raise


def predict_with_probabilities(text, model, top_n=10):
    vectorized_text = clean_and_vectorize(text)
    predictions_proba = model.predict_proba(vectorized_text)
    top_n_indices = np.argsort(predictions_proba[0])[-top_n:][::-1]
    top_n_categories = model.classes_[top_n_indices]
    top_n_probs = predictions_proba[0][top_n_indices]
    return top_n_categories, top_n_probs

def classify_input(input_text):
    try:
        vec_text = clean_and_vectorize(input_text)
        category_prediction = category_model.predict(vec_text)[0]
        subcategory_prediction = subcategory_model.predict(vec_text)[0]
        
        category_prob = max(category_model.predict_proba(vec_text)[0])
        subcategory_prob = max(subcategory_model.predict_proba(vec_text)[0])
        
        logger.info(f"Classification completed successfully for text: {input_text}")
        return category_prediction, subcategory_prediction, category_prob, subcategory_prob
    except Exception as e:
        logger.error(f"Error in classify_input: {e}")
        raise

def classify_and_map(input_text):
    try:
        # Predict category
        cat_predictions, cat_probs = predict_with_probabilities(input_text, category_model)
        # Predict subcategory
        subcat_predictions, subcat_probs = predict_with_probabilities(input_text, subcategory_model)

        # Determine valid subcategories with highest probability
        top_category = cat_predictions[0]
        valid_subcategories = cat_subcat_map.get(top_category, {})
        highest_prob = 0
        best_subcat = None
        for subcat, prob in zip(subcat_predictions, subcat_probs):
            if subcat in valid_subcategories and prob > highest_prob:
                highest_prob = prob
                best_subcat = subcat

        if best_subcat:
            subcategory = best_subcat
            subcategory_prob = highest_prob
        else:
            return None, None, None, None

        category = top_category
        category_prob = cat_probs[0]
        
        logger.info(f"Classification and mapping completed successfully for text: {input_text}")
        return category, subcategory, category_prob, subcategory_prob
    except Exception as e:
        logger.error(f"Error in classify_and_map: {e}")
        raise

def retrain_model(new_data, new_category, new_subcategory):
    try:
        # Log the received inputs
        logger.info(f"Received new data: {new_data}, new category: {new_category}, new subcategory: {new_subcategory}")
        
        # Dummy operation
        logger.info("Retraning Done.")
        
        return True
    except Exception as e:
        logger.error(f"Error in retrain_model: {e}")
        raise


def classify_items(items):
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
                'category': None,
                'subcategory': None,
                'message': "Description Missing"
            })
            continue
        try:
            category, subcategory, category_prob, subcategory_prob = classify_and_map(item_description)
            
            if category is None or subcategory is None:
                results.append({
                    'itemNumber': item_number,
                    'itemDescription': item_description,
                    'message': "No Valid Prediction"
                })
                continue
            
            classified_item = {
                'itemNumber': item_number,
                'itemDescription': item_description,
                'category': category.strip(),
                'subcategory': subcategory.strip(),
                'category_probability': category_prob,
                'subcategory_probability': subcategory_prob
            }
            results.append(classified_item)
        except Exception as e:
            errors.append(f"Error processing item {item_number}: {str(e)}")
    return results, errors

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
        new_entry = pd.DataFrame({'Description': [clean_text(new_data)], 'Category': [new_category], 'SubCategory': [new_subcategory]})
        df = pd.concat([df, new_entry], ignore_index=True)
        df['CombinedCategory'] = df['Category'].str.cat(df['SubCategory'], sep='-')
        df.to_excel('TrainingData-NB-Balanced-New.xlsx', index=False)
        
        if retrain_attempts[key] >= 3:
            X_train = df['Description']
            y_train = df['CombinedCategory']
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

