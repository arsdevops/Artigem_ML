from fastapi import FastAPI, Form, HTTPException, status
from fastapi.responses import JSONResponse
from typing import List
import uvicorn
import logging
import text_classification_new
from wrapper import extract_all_entities
from fastapi.middleware.cors import CORSMiddleware 

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

@app.post("/predict")
async def predict_category_endpoint(input_text: str = Form(default="")):
    input_text = input_text.strip()
    if input_text == "":
        logger.error("Received empty or whitespace-only as input text.")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Input text is empty or only whitespace."
        )
    try:
        category, subcategory, category_prob, subcategory_prob = text_classification_new.classify_and_map(input_text)
        
        if category is None or subcategory is None:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"message": "No Valid Prediction"}
            )

        logger.info(f"Prediction completed successfully for text: {input_text}")
        return JSONResponse(status_code=status.HTTP_200_OK, content={
            "category": category, 
            "subcategory": subcategory, 
            "category_probability": category_prob,
            "subcategory_probability": subcategory_prob
        })

    except HTTPException as e:
        logger.error(f"HTTPException in /predict: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Error in /predict: {e}")
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"detail": str(e)})

@app.post("/classify-items")
async def classify_items_endpoint(items: List[dict]):
    try:
        processed_items, errors = text_classification_new.classify_items(items)
        
        if errors:
            logger.error("Errors encountered during classification.")
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={"errors": errors, "processed_items": processed_items}
            )
        logger.info("Items classified successfully.")
        return JSONResponse(status_code=status.HTTP_200_OK, content=processed_items)
    except Exception as e:
        logger.error(f"Error in /classify-items: {e}")
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"detail": str(e)})

@app.post("/retrain")
async def retrain_model_endpoint(new_data: str = Form(...), new_category: str = Form(...), new_subcategory: str = Form(...)):
    try:
        text_classification_new.retrain_model(new_data, new_category, new_subcategory)
        logger.info("Model retrained successfully (dummy operation).")
        return JSONResponse(status_code=status.HTTP_200_OK, content={"message": "Model Retrain on the given data initiated."})
    except Exception as e:
        logger.error(f"Error in /retrain: {e}")
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"detail": str(e)})

@app.post("/extract-entities/")
async def extract_entities_endpoint(item_description: str = Form(...)):
    item_description = item_description.strip()
    if not item_description:
        logger.error("Received empty or whitespace-only as item description.")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Item description is empty or only whitespace."
        )

    try:
        # Call the new function to extract all entities
        entities = extract_all_entities(item_description)
        if not entities or all(value is None for value in entities.values()):
            logger.info("No entities extracted for the given description.")
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"message": "No entities found for the given description."}
            )

        logger.info("Entities extracted successfully.")
        return JSONResponse(status_code=status.HTTP_200_OK, content={"entities": entities})

    except Exception as e:
        logger.error(f"Error in /extract-entities/ endpoint: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": str(e)}
        )

@app.post("/predict_nb")
async def predict_category_endpoint_nb(input_text: str = Form(default="")):
    input_text = input_text.strip()
    if input_text == "":
        logger.error("Received empty or whitespace-only as input text.")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Input text is empty or only whitespace."
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
    try:
        processed_items, errors = text_classification_new.classify_items_nb(items)
        if errors:
            logger.error(f"Errors encountered during classification: {errors}")
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={"errors": errors, "processed_items": processed_items}
            )
        logger.info("Items classified successfully.")
        return JSONResponse(status_code=status.HTTP_200_OK, content=processed_items)
    except Exception as e:
        logger.error(f"Error in /classify-items_nb: {e}")
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"detail": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)

