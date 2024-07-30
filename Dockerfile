FROM python:3.10

WORKDIR /app

COPY . /app

# Install dependencies
RUN pip install --upgrade pip && \
    pip install numpy && \
    pip install scikit-learn==1.1.3 && \
    pip install fastapi uvicorn pandas nltk python-multipart joblib spacy word2number openpyxl

# Download NLTK data
RUN python3 -m nltk.downloader wordnet

# Download SpaCy model
RUN python -m spacy download en_core_web_sm

EXPOSE 9080

CMD ["python3", "app.py"]

