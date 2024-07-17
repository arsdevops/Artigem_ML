import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import random
import os
import json
from config_reader import config

MODEL_PATH = config['model']['model_path']
TRAINING_DATA_PATH = config['data']['training_data_json']

def load_training_data(data_path=TRAINING_DATA_PATH):
    with open(data_path, 'r') as file:
        data = json.load(file)
    return [(item['text'], {'entities': [(ent['start'], ent['end'], ent['label']) for ent in item['entities']]}) for item in data]

def train_ner_model(model_path=MODEL_PATH, iterations=30):
    TRAIN_DATA = load_training_data()

    if not os.path.exists(model_path):
        os.makedirs(model_path)
        print(f"Created directory {model_path}")

    nlp = spacy.load("en_core_web_lg")
    ner = nlp.create_pipe("ner") if "ner" not in nlp.pipe_names else nlp.get_pipe("ner")

    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    if "ner" not in nlp.pipe_names:
        nlp.add_pipe(ner, last=True)

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            random.shuffle(TRAIN_DATA)
            losses = {}
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                examples = [Example.from_dict(nlp.make_doc(text), annotations) for text, annotations in zip(texts, annotations)]
                nlp.update(examples, drop=0.5, losses=losses)
            print(f"Iteration {itn}, Losses: {losses}")

    nlp.to_disk(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    try:
        print("Starting model training...")
        train_ner_model()
        print("Model training completed.")
    except Exception as e:
        print(f"Failed to train and save model: {e}")

