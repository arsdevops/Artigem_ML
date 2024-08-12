from fastapi import FastAPI, Form, HTTPException, status
from fastapi.responses import JSONResponse
from typing import List
import uvicorn
import logging
import text_classification_new
from wrapper import extract_all_entities
from fastapi.middleware.cors import CORSMiddleware 
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

logging.basicConfig(filename='api.log', level=logging.INFO, format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')
logger = logging.getLogger(__name__)

def is_gibberish(input_text):
    tokens = re.findall(r'\b\w+\b', input_text)
    gibberish_tokens = 0
    for token in tokens:
        if len(token) > 1 and len(re.findall(r'[^a-zA-Z]', token)) > len(token) / 2:
            gibberish_tokens += 1
    if gibberish_tokens / len(tokens) > 0.5:
        return True
    if any(re.search(r'(.)\1{3,}', token) for token in tokens):
        return True

    return False

@app.post("/predict_nb")
async def predict_category_endpoint_nb(input_text: str = Form(default="")):
    input_text = input_text.strip()
    if input_text == "":
        logger.error("Received empty or whitespace-only as input text.")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Input text is empty or only whitespace."
        )
    
    # Check for gibberish input
    if is_gibberish(input_text):
        logger.info("Gibberish input detected. Model will not make a prediction.")
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": "The model is not confident in its prediction for this input."}
        )

    try:
        _, probabilities, class_labels = text_classification_new.classify_input_nb(input_text)
        label_probability_pairs = sorted(zip(class_labels, probabilities), key=lambda x: x[1], reverse=True)
        
        max_prob_class_label, max_prob = label_probability_pairs[0]
        parts = max_prob_class_label.split('-')
        category = parts[0]
        subcategory = '-'.join(parts[1:]) if len(parts) > 1 else ''

        if max_prob <= 0.10:
            return JSONResponse(
                status_code=status.HTTP_200_OK, 
                content={"message": "The model is not confident in its prediction for this input."}
            )
        logger.info("Prediction completed successfully.")
        return JSONResponse(status_code=status.HTTP_200_OK, content={
            "category": category, 
            "subcategory": subcategory, 
            "probability": max_prob
        })

    except HTTPException as e:
        logger.error(f"HTTPException in /predict: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Error in /predict: {e}")
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"detail": str(e)})


@app.post("/retrain_nb")
async def retrain_model_endpoint_nb(new_data: str = Form(...), new_category: str = Form(...), new_subcategory: str = Form(...)):
    try:
        message = text_classification_new.retrain_model_nb(new_data, new_category, new_subcategory)
        logger.info(message)
        return JSONResponse(status_code=status.HTTP_200_OK, content={"message": message})
    except Exception as e:
        logger.error(f"Error in /retrain_nb: {e}")
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"detail": str(e)})

@app.post("/classify-items_nb")
async def classify_items_endpoint_nb(items: List[dict]):
    if not items:
        logger.error("Received an empty list for classification.")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The item list for classification is empty."
        )

    processed_items = []
    errors = []

    for item in items:
        item_description = item.get("itemDescription", "").strip()

        # Check if the item description is gibberish
        if is_gibberish(item_description):
            logger.info(f"Gibberish input detected: {item_description}")
            processed_items.append({
                "itemNumber": item.get("itemNumber"),
                "itemDescription": item_description,
                "message": "The model is not confident in its prediction for this input."
            })
            continue  # Skip classification for gibberish items

        logger.info(f"Non-gibberish input: {item_description}")

        # Classify the non-gibberish input (mimic predict_nb logic)
        try:
            _, probabilities, class_labels = text_classification_new.classify_input_nb(item_description)
            label_probability_pairs = sorted(zip(class_labels, probabilities), key=lambda x: x[1], reverse=True)
            
            max_prob_class_label, max_prob = label_probability_pairs[0]
            parts = max_prob_class_label.split('-')
            category = parts[0]
            subcategory = '-'.join(parts[1:]) if len(parts) > 1 else ''

            if max_prob <= 0.10:
                processed_items.append({
                    "itemNumber": item.get("itemNumber"),
                    "itemDescription": item_description,
                    "message": "The model is not confident in its prediction for this input."
                })
            else:
                processed_items.append({
                    "itemNumber": item.get("itemNumber"),
                    "itemDescription": item_description,
                    "category": category,
                    "subcategory": subcategory,
                    "probability": max_prob
                })

        except Exception as e:
            logger.error(f"Error during classification of {item_description}: {e}")
            processed_items.append({
                "itemNumber": item.get("itemNumber"),
                "itemDescription": item_description,
                "error": str(e)
            })

    logger.info("Items classified successfully.")
    return JSONResponse(status_code=status.HTTP_200_OK, content=processed_items)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9080)

