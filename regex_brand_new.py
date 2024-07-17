import pandas as pd
import re
from config_reader import config

# Load configuration for file paths
brand_list_excel = config['files']['brand_list_excel']
brand_patterns_text = config['files']['brand_patterns_text']

# Load brand names from Excel
df = pd.read_excel(brand_list_excel)
brand_names = df['Retailer'].unique()

def create_advanced_structured_pattern(brand_names):
    pattern_parts = []
    for name in brand_names:
        if isinstance(name, str):
            escaped_name = re.escape(name)
            pattern = "\\b" + escaped_name.replace("\\ ", "\\s*") + "(\\B|\\b)"
            pattern_parts.append(pattern)

    return '|'.join(sorted(set(pattern_parts), key=len, reverse=True))

def dynamic_post_processing(matches):
    return list(set(matches))

def extract_complete_brands(text, regex_pattern, brand_names):
    matches = re.finditer(regex_pattern, text, re.IGNORECASE)
    matched_brands = []
    for match in matches:
        match_str = match.group()
        for brand in brand_names:
            if isinstance(brand, str) and brand.lower().replace(' ', '') in match_str.lower().replace(' ', ''):
                matched_brands.append(brand)
                break

    return dynamic_post_processing(matched_brands)

advanced_regex_pattern = create_advanced_structured_pattern(brand_names)

# Save the pattern to a file
with open(brand_patterns_text, 'w') as file:
    file.write(advanced_regex_pattern)

print(f"The regex pattern has been saved to {brand_patterns_text}.")



texts = [
    "Save-Rite Drugs RX 1188069 00 Brinkley Brahm 7.5ml Oseltamivir 6mg Qty  75 (medical insurance paid)",
    "HillmanWood Screws Flat HeadPhillips and lily & DanSize or SimpleLotion, Time and TruSize",
    "I recently shopped at ZWILLINGand SKECHERSShoes and then from AmazonTV, Bath & Body WorksLotion, Hp, Samsung and Kroger, lily & Dan, K-Swiss  "
]

for text in texts:
    extracted_brands = extract_complete_brands(text, advanced_regex_pattern, brand_names)
    print(extracted_brands)

