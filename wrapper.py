import pandas as pd
from extracted_entities import extract_entities as ner_extract_entities
import regex_brand_new
from regex_brand_new import extract_complete_brands
from config_reader import config

# Load configuration for file paths
brand_list_excel = config['files']['brand_list_excel']
brand_patterns_text = config['files']['brand_patterns_text']

# Load brand names from Excel
df = pd.read_excel(brand_list_excel)
brand_names = df['Retailer'].unique()

# Load regex patterns from text file
with open(brand_patterns_text, 'r') as file:
    advanced_regex_pattern = file.read()

def extract_all_entities(item_description):
    # Extract brands using the custom regex approach with brand name filtering
    extracted_brands_regex = extract_complete_brands(item_description, advanced_regex_pattern, brand_names)

    # Extract entities using the trained NER model
    entities = ner_extract_entities(item_description)

    # Prefer the brand extracted by the regex if available
    if extracted_brands_regex:
        entities['brand'] = extracted_brands_regex[0]  # Taking the first brand found as the primary one

    return entities

if __name__ == "__main__":
    item_description = "4 Dior extra hold hair spray 2 fl oz with 6 month warranty for $25  "
    extracted_entities = extract_all_entities(item_description)
    print(extracted_entities)

