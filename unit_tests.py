
import pytest
from flask import Flask
from sentiment_analysis import authenticate
from unittest.mock import patch
import os
from sentiment_analysis import sentiment_analysis


@pytest.fixture
def client():
    app = Flask(__name__)
    app.config['TESTING'] = True
    os.environ['AUTH_TOKEN'] = '1234'
    os.environ['OPENAI_API_KEY'] = 'your_api_key'

    @app.route('/test')
    @authenticate
    def test_route():
        return "Success", 200

    app.route('/sentiment', methods=['POST'])(authenticate(sentiment_analysis))

    with app.test_client() as client:
        yield client
    # Clean up environment variables
    del os.environ['AUTH_TOKEN']


# Example setup to ensure AUTH_TOKEN matches the test token value
def test_token_required_with_valid_token(client):
    headers = {'Authorization': 'Bearer 1234'}
    response = client.get('/test', headers=headers)
    assert response.status_code == 200


def test_token_required_with_invalid_token(client):
    headers = {'Authorization': 'Bearer 0000'}
    response = client.get('/test', headers=headers)
    assert response.status_code == 401


def test_sentiment_analysis_with_valid_text(client):
    # Set up the request headers including the Authorization header with a valid token
    headers = {
        'Authorization': 'Bearer 1234',
        'Content-Type': 'application/json'
    }
    # Define the payload for the POST request, which includes the text to analyze
    payload = {
        "text": "I love Virgin Airlines!"
    }
    # Make the POST request to the /sentiment endpoint with the payload and headers
    response = client.post('/sentiment', json=payload, headers=headers)

    # Assert that the status code of the response is 200, indicating success
    assert response.status_code == 200


def test_sentiment_analysis_with_invalid_text(client):
    # Set up the request headers including the Authorization header with a valid token
    headers = {
        'Authorization': 'Bearer 1234',
        'Content-Type': 'application/json'
    }

    # Test with empty text
    response = client.post('/sentiment', json={'text': ''}, headers=headers)
    assert response.status_code == 400
    # Assuming the response includes a specific error message for empty text,
    # validate that the expected message is part of the response
    assert 'No text provided' in response.get_data(as_text=True), "Expected error message for empty text not found"

    # Test with missing text field
    response = client.post('/sentiment', json={}, headers=headers)
    assert response.status_code == 400

    assert 'No text provided' in response.get_data(as_text=True), "Expected error message for missing text not found"


def test_sentiment_analysis_with_fallback_to_custom_model(client):
    # Set up the request headers including the Authorization header with a valid token
    headers = {
        'Authorization': 'Bearer 1234',
        'Content-Type': 'application/json'
    }
    # Define the payload for the POST request, which includes the text to analyze
    payload = {
        "text": "Testing the fallback scenario."
    }

    with patch('sentiment_analysis.openai_sentiment_analysis', side_effect=Exception("Simulated API failure")):
        # Make the POST request to the /sentiment endpoint with the payload and headers
        response = client.post('/sentiment', json=payload, headers=headers)

        # Assert that the status code of the response is 200, indicating success
        assert response.status_code == 200, "Expected a successful response status code"

        # Assert the expected response from your custom model
        response_data = response.get_json()
        assert response_data.get('sentiment_value') is not None, "Expected a sentiment score in the response"
