# Sentiment Analysis Flask Application
This Flask application provides an API endpoint for sentiment analysis, utilizing both a local machine learning model through the transformers library and the OpenAI API. Depending on the input and operational conditions, the application chooses the most suitable method for sentiment analysis and returns the result in JSON format.

# Prerequisites

Before running this application, ensure you have the following installed:

Python 3.8 or higher
pip (Python package installer)
This application also requires an OpenAI API key set as an environment variable. 


# Setup
###  1. Clone the Repository
Clone this repository to your local machine using git, or download the source code directly.

### 2. Create a Conda Environment
It's recommended to create a Conda environment to manage the application's dependencies:

```
conda create --name sentiment_analysis python=3.8
conda activate sentiment_analysis
```

### 3. Install Dependencies
Dependencies are listed in a requirements.txt file. Install them using:


```
pip install -r requirements.txt
```


### 4. Configure Environment Variables
Configure the following environment variables:

OPENAI_API_KEY: Your OpenAI API key.
AUTH_TOKEN: A custom authentication token for accessing the API. 

### 5. Get Data (optional)
If you would like to run fine-tuning on Twitter US Airline Sentiment.

Download the dataset and create a folder under the project called `/data` and store the file there. 


# Running the Application
### Start the Application
With the environment variable configured and dependencies installed, start the application with:

```
python sentiment_analysis.py
```


Ensure you're in the activated Conda environment and in the directory containing app.py.

### Accessing the API
The application exposes a /sentiment endpoint for POST requests. Requests should include a JSON body with the text for analysis. Example:

### Test the API using curl with your authentication token:

```
curl -X POST http://localhost:5000/sentiment \
-H "Content-Type: application/json" \
-H "Authorization: Bearer 1234" \
-d "{\"text\":\"Your text here\", \"airline_sentiment\": \"neutral\", \"airline_sentiment_confidence\": 1.0, \"negativereason\": \"\", \"negativereason_confidence\": \"\", \"airline\": \"Virgin America\", \"airline_sentiment_gold\": \"\", \"name\": \"cairdin\", \"negativereason_gold\": \"\", \"retweet_count\": 0, \"tweet_coord\": \"\", \"tweet_created\": \"2015-02-24 11:35:52 -0800\", \"tweet_location\": \"\", \"user_timezone\": \"Eastern Time (US & Canada)\"}"

```
Replace YOUR_TOKEN with the AUTH_TOKEN you configured.

Depending on your terminal your terminal may not process the JSON correctly. 

```
curl -X POST http://localhost:5000/sentiment \
-H "Content-Type: application/json" \
-H "Authorization: Bearer 1234" \
-d '{"text":"Your text here", "airline_sentiment": "neutral", "airline_sentiment_confidence": 1.0, "negativereason": "", "negativereason_confidence": "", "airline": "Virgin America", "airline_sentiment_gold": "", "name": "cairdin", "negativereason_gold": "", "retweet_count": 0, "tweet_coord": "", "tweet_created": "2015-02-24 11:35:52 -0800", "tweet_location": "", "user_timezone": "Eastern Time (US & Canada)"}'
```

# Additional Application Components

## Fine-Tuning Models
### Purpose
The `train.py` script is designed  to fine-tune the sentiment analysis model further to suit Twitter US Airline Sentiment dataset. This script uses the transformers library to adjust the model parameters based on new training data.

### How to Use
Prepare Your Dataset: Ensure your dataset is in a CSV format with at least two columns: one for the text and another for its corresponding sentiment. The script is configured to read from a file named Tweets.csv by default.

Dataset Format: The sentiment values in your dataset should be categorized as 'negative', 'neutral', or 'positive'. The script automatically maps these to numeric values (-1, 0, and 1, respectively) for regression purposes.

Running the Script: Activate the sentiment_analysis Conda environment. Navigate to the script's directory and run:
```
python train.py
```

Ensure your dataset is placed correctly as per the script's expected path (default is data/Tweets.csv).

Customization: You can modify the script to point to a different dataset or adjust the training parameters and model configuration as needed.

If you would like to use this with `/sentiment` API endpoint the fine-tuned model weights need to be loaded in the flask application environment. 

```
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
```

`trained` needs to be set as `True`


### Script Details
Model Initialization: The script initializes a custom PyTorch model that wraps the pre-trained RoBERTa model with an added regression head.
Dataset Handling: Custom Dataset classes handle training and validation data, managing tokenization and encoding of text.
Training: A custom Trainer class fine-tunes the model using the specified training arguments, employing a regression approach suitable for sentiment analysis.
Saving the Model: After training, the model is saved to ./results/trained_roberta_regression, allowing for later use in sentiment analysis tasks.

# Integrating OpenAI Embeddings with RandomForestRegressor

If full parameter fine-tuning using an open source model from huggingface is not suitable for the data. One can explore a number of other opportunities:
1. [OPENAI Embedding API](https://platform.openai.com/docs/guides/embeddings/use-cases) for feature encoding and fine-tuning a light weight machine learning model e.g. [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor.predict).

This workflow demonstrates a powerful approach to handling textual data for sentiment analysis, leveraging the capabilities of OpenAI's Embeddings API in conjunction with a RandomForestRegressor model. It allows for the efficient processing of text data into a format suitable for machine learning, enabling the development of predictive models with a high degree of accuracy.

`sklearn_train.py` is used for training the machine learning model and the model is saved onto disk and can be used at inference time in `sentiment_analysis.py`

# Unit Tests
Overview
The `unit_tests.py` file contains a suite of tests designed to validate the functionality and robustness of the sentiment_analysis.py module within a Flask application. This module provides an API for performing sentiment analysis on text. The tests ensure that:

1. The API authentication mechanism works as expected.
2. The sentiment analysis endpoint correctly processes valid and invalid requests.
3. The system gracefully handles errors, such as external api failures by falling back to a custom model.

### Setting Up
Before running the tests, ensure that the Flask environment is correctly set up and that all necessary environment variables are defined. Specifically, AUTH_TOKEN should be set to a test value ('1234' is used in the tests) and OPENAI_API_KEY should be set to your OpenAI API key.

Running the Tests
To run the tests, execute the following command in your terminal:

```
pytest unit_tests.py::name_of_unit_test
```

```
pytest unit_tests.py::test_token_required_with_valid_token
```

# Future Work
### Learning processes
1. Optionally one can use a pretrained open source languae model as the feature encoder instead of openai embedding api to reduce costs.


### Evaluation metrics
1. For regression tasks where they can be interpreted as a classification e.g. regression for sentiment analysis one can use traditional classification metrics
   1. Binned Accuracy: Make sure to define a threshold for the sentiment scores so that they belong to a particular category. For example [-1, -0.33] corresponds with negative [-0.33,0.33] corresponds to neutral and [0.33,1] corresponds to positive.
```
    def bin_predictions(predicted_scores, thresholds=(-1/3, 1/3)):
        binned_predictions = np.zeros(predicted_scores.shape)
        binned_predictions[predicted_scores <= thresholds[0]] = -1
        binned_predictions[predicted_scores > thresholds[1]] = 1
        # Scores between thresholds[0] and thresholds[1] are left as 0 (neutral), no need to explicitly set them
        return binned_predictions

    def calculate_accuracy(actual_labels, predicted_labels):
        correct_predictions = np.sum(actual_labels == predicted_labels)
        total_predictions = len(actual_labels)
        accuracy = correct_predictions / total_predictions
        return accuracy
        
    # Bin the predicted scores
    binned_predictions = bin_predictions(predicted_scores)
    
    # Calculate the accuracy
    accuracy = calculate_accuracy(actual_labels, binned_predictions)
    
    print(f"Accuracy: {accuracy}")
      
```
2. I noticed the dataset is quite imbalanced, in this case, using a confusion matrix to understand which classes are being confused with others is going to be very helpful to improve model performance.
```
negative    9178
neutral     3099
positive    2363
Name: airline_sentiment, dtype: int64
```

### Confidence scores

Log probabilities can be interpreted as confidence values and within the openai text generation output you can have the model return the corresponding log probabilities.

