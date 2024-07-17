from word2number import w2n
import re

def post_process_quantities(quantities_extracted, item_description):
    multiplication_symbols = ['×', '*', 'x']
    for symbol in multiplication_symbols:
        multiplication_regex = r'(\d+|[a-zA-Z]+)\s*[\*x×]\s*(\d+|[a-zA-Z]+)'
        matches = re.findall(multiplication_regex, item_description)
        if matches:
            for match in matches:
                try:
                    number_after = match[1]
                    if number_after.isdigit():
                        return int(number_after)
                    return w2n.word_to_num(number_after)  
                except ValueError:
                    continue
    for quantity in quantities_extracted:
        try:
            if quantity.isdigit():
                return int(quantity)
            return w2n.word_to_num(quantity)  
        except ValueError:
            continue
    return 1 

def convert_price(entity_text):
    normalized_text = re.sub(r'\bdollars?\b', '', entity_text, flags=re.IGNORECASE).strip()
    normalized_text = re.sub(r',', '', normalized_text)  # Remove commas for simplicity
    special_case_normalized = re.sub(r'\bthousand hundred\b', '1100', normalized_text, flags=re.IGNORECASE)

    try:
        return f"${w2n.word_to_num(special_case_normalized):.2f}"
    except ValueError:
        pass  

    mixed_numeric_pattern = re.compile(r'(\d+)\s*(thousand|hundred)\b|\b(hundred|thousand)\s*\$(\d+(\.\d{2})?)', re.IGNORECASE)
    for match in mixed_numeric_pattern.finditer(normalized_text):
        num, mag, word, amount, _ = match.groups()
        if num and mag:  
            multiplier = 1000 if mag.lower() == "thousand" else 100
            return f"${int(num) * multiplier:.2f}"
        elif word and amount: 
            multiplier = 1000 if word.lower() == "thousand" else 100
            total = multiplier + float(amount)
            return f"${total:.2f}"

    direct_numeric_match = re.search(r'\$?(\d+(\.\d{2})?)', normalized_text)
    if direct_numeric_match:
        numeric_value = direct_numeric_match.group(1)
        return f"${float(numeric_value):,.2f}"

    try:
        amount = w2n.word_to_num(normalized_text)
        return f"${amount:.2f}"
    except ValueError:
        pass  
    return entity_text

def convert_word_to_number(entity_text):
    try:
        return w2n.word_to_num(entity_text)
    except ValueError:
        words = entity_text.split()
        converted_words = []
        for word in words:
            try:
                converted_words.append(str(w2n.word_to_num(word)))
            except ValueError:
                converted_words.append(word)
        return " ".join(converted_words)

