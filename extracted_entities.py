import spacy
from ner_utils import post_process_quantities, convert_word_to_number, convert_price
import re
from config_reader import config

MODEL_PATH = config['model']['model_path']

def extract_entities(item_description):
    nlp = spacy.load(MODEL_PATH)
    doc = nlp(item_description)
    entities = {
        "quantity": 1,  
        "price": None,
        "item_age_years": None,
        "item_age_months": None,
        "brand": None,
    }

    isbn_pattern = re.compile(r'\b97[89]-?\d{1,5}-?\d{1,7}-?\d{1,7}-?[0-9X]\b')

    quantities_extracted = []
    for ent in doc.ents:
        if ent.label_ == "QUANTITY":
            quantities_extracted.append(ent.text)
        elif ent.label_ == "PRICE":
            # Check if the extracted price is actually an ISBN
            if not isbn_pattern.search(ent.text):
                converted_price = convert_price(ent.text)
                entities["price"] = converted_price
            else:
                print(f"Skipped ISBN as Price: {ent.text}")
        elif ent.label_ == "ITEM_AGE(Years)":
            entities["item_age_years"] = convert_word_to_number(ent.text)
        elif ent.label_ == "ITEM_AGE(Months)":
            entities["item_age_months"] = convert_word_to_number(ent.text)
        elif ent.label_ == "BRAND":
            entities["brand"] = ent.text

    processed_quantities = post_process_quantities(quantities_extracted, item_description)
    entities["quantity"] = processed_quantities

    return entities

if __name__ == "__main__":
    text = "4 Dior extra hold hair spray 2 fl oz with 6 month warranty for $25 "
    print(extract_entities(text))

