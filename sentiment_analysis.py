import pickle
import os
import torch
from flask import Flask, request, jsonify
from functools import wraps
import openai
import json
from train import CustomRegressionModel
from transformers import AutoModel
from transformers import AutoTokenizer
import logging
from logging.handlers import RotatingFileHandler
import re

app = Flask(__name__)

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Validate essential environment variables
client = None
if OPENAI_API_KEY:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
else:
    app.logger.warning("OpenAI API key is missing. Ensure OPENAI_API_KEY is set. Some functionality may be limited.")

# Setup logging
if not os.path.exists('logs'):
    os.mkdir('logs')
file_handler = RotatingFileHandler('logs/api.log', maxBytes=10240, backupCount=10)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
file_handler.setLevel(logging.INFO)
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)
app.logger.info('API startup')


def authenticate(f):
    """
    Decorator to enforce token authentication on routes.

    Parameters:
    - f (function): The function to decorate.

    Returns:
    - function: The wrapped function with authentication.
    """

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if app.config['TESTING']:
            return f(*args, **kwargs)
        AUTH_TOKEN = os.getenv("AUTH_TOKEN")
        auth_header = request.headers.get('Authorization')
        token = auth_header.split(" ")[1] if auth_header and auth_header.startswith('Bearer ') else None
        if not token or token != AUTH_TOKEN:
            return jsonify({"error": "Access token is missing or invalid"}), 401
        return f(*args, **kwargs)

    return decorated_function


def load_model(trained=False, model_path=None):
    """
    Loads the custom sentiment analysis model from a pre-trained base model or fine-tuned model.

    Returns:
    - model: The loaded and pre-configured model ready for inference.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = AutoModel.from_pretrained(MODEL_NAME)
    model = CustomRegressionModel(base_model).to(device).eval()
    if trained:
        model.load_state_dict(torch.load(model_path))
    return model


def load_tokenizer():
    """
    Loads the tokenizer associated with the sentiment analysis model.

    Returns:
    - tokenizer: The tokenizer for processing text input.
    """
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def analyze_sentiment_custom(text):
    """
    Performs sentiment analysis using the custom model.

    Parameters:
    - text (str): The input text to analyze.

    Returns:
    - float: The sentiment value as a floating-point number.
    """

    model = load_model()
    tokenizer = load_tokenizer()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        prediction = model(**inputs)
    sentiment_score = float(prediction.cpu().numpy()[0][0])
    return sentiment_score


def openai_sentiment_analysis(text):
    """
    Performs sentiment analysis using an openai model and fallbacks to custom language model from huggingface.

    Parameters:
    - text (str): The input text to analyze.

    Returns:
    - floatsentiment_value = analyze_sentiment_custom(text): The sentiment value as a floating-point number.
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",
             "content": "Given the text: {}, provide a sentiment rating on a scale from -1 to 1, where -1 "
                        "represents extremely negative sentiment, 0 is neutral, and 1 represents extremely "
                        "positive sentiment. Respond with a numeric value only.".format(
                 text)},
        ],
        seed=2024
    )
    sentiment_value = response.choices[0].message.content
    return sentiment_value


def openai_sentiment_analysis_embeddings(text):
    """
    Performs sentiment analysis using an openai embedding api with a trained lightweight regressor model and
    fallbacks to custom language model from huggingface.

    Parameters:
    - text (str): The input text to analyze.

    Returns:
    - float: The sentiment value as a floating-point number.
    """
    try:
        def get_embedding(text, model="text-embedding-3-small"):
            text = text.replace("\n", " ")
            return client.embeddings.create(input=[text], model=model).data[0].embedding

        embeddings = get_embedding(text)
        # Load the RandomForestRegressor model from a pickle file
        with open('./random_forest_model.pkl', 'rb') as model_file:
            rfr = pickle.load(model_file)
        # Ensure embeddings is a 2D array for prediction
        sentiment_value = rfr.predict([embeddings]).item()
        print(sentiment_value)
    except Exception as external_api_error:
        app.logger.error(f"Embedding analysis failed: {str(external_api_error)}")
        sentiment_value = analyze_sentiment_custom(text)
    return sentiment_value


def preprocess_text(text):
    """
    Clean the input text for model inference by removing URLs, and performing
    basic text normalization like trimming and lowering the case.

    Parameters:
    - text (str): The text to be cleaned.

    Returns:
    - str: The preprocessed text.
    """
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)

    text = text.lower().strip()

    return text


@app.route('/sentiment', methods=['POST'])
@authenticate
def sentiment_analysis():
    """
    Endpoint for sentiment analysis. It uses OpenAI by default and falls back to the custom model on failure.

    Parameters:
    - None directly, but expects a JSON payload with "text" field in POST request.

    Returns:
    - JSON response: Contains either the sentiment value or an error message.
    """
    data = request.json
    text = data.get("text")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    text = preprocess_text(text)

    try:
        sentiment_value = openai_sentiment_analysis(text)
        return jsonify({"sentiment_value": sentiment_value}), 200
    except Exception as e:
        app.logger.error(f"OpenAI sentiment analysis failed: {str(e)}")
        try:
            sentiment_value = analyze_sentiment_custom(text)
            return jsonify({"sentiment_value": sentiment_value}), 200
        except Exception as e:
            app.logger.error(f"Custom LLM sentiment analysis failed: {str(e)}")


@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f"Server Error: {error}")
    return jsonify({"error": "An unexpected error occurred"}), 500


@app.errorhandler(404)
def not_found_error(error):
    app.logger.error(f"Not Found: {error}")
    return jsonify({"error": "Resource not found"}), 404


if __name__ == '__main__':
    app.run(debug=True)
