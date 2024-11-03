from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flasgger import Swagger
from flask_cors import CORS
import joblib
import numpy as np
import os
import logging
from logging.handlers import RotatingFileHandler
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from collections import defaultdict
import requests

app = Flask(__name__)

# Configure CORS to allow cross-origin requests
CORS(app)
CORS(app, resources={r"/": {"origins": "*"}})  # Adjust the origins as needed

# Configure Swagger
swagger_config = {
    "definitions": {"name": "Sparta 300"},
    'swagger': '2.0',
    'info': {
        'version': '0.0.1',
        'title': 'Campaign Prediction API',
        'description': "This API provides the prediction API which you can predict the Raised Amount",
        'termsOfService': '/tos'
    },
    'specs': [
        {
            'endpoint': 'apispec',
            'route': '/apispec.json',
            'rule_filter': lambda rule: True,
            'model_filter': lambda tag: True,
        }
    ],
    'headers': [],  # Set headers to an empty list to avoid TypeError
}
swagger = Swagger(app, config=swagger_config)

# Use an application context to access current_app
with app.app_context():
    print(swagger.get_apispecs(endpoint='apispec'))

# Set up rate limiting (100 requests per minute per IP)
limiter = Limiter(get_remote_address, app=app, default_limits=["100 per minute"])


# Configure Logging
def setup_logging():
    logging.basicConfig(level=logging.INFO)
    handler = RotatingFileHandler("app.log", maxBytes=2000, backupCount=5)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)


setup_logging()
logging.info("Logging is set up.")

# Load pre-trained models
model1 = joblib.load('Campaign_prediction.joblib')


# apidocs
@app.route('/custom_swagger', methods=['GET'])
@limiter.limit("100 per minute")
def swagger_json():
    api_spec = swagger.get_apispecs(endpoint='apispec')
    print(api_spec)  # Debug print
    print("swagger.json endpoint was hit")
    return api_spec


# Endpoint for delay prediction
@app.route('/campaign_predict', methods=['POST'])
@limiter.limit("100 per minute")
async def prediction1():
    """
    Predicts Raised Amount of any campaign.
    ---
    parameters:
      - name: features
        in: body
        schema:
          type: array
          items:
            type: number
        required: true
        description: Features for delay prediction.
    responses:
      200:
        description: Raised Amount Prediction
      500:
        description: Internal server error
    """
    try:
        data = request.json
        features = np.array(data['features'])
        prediction = model1.predict([features])[0]

        feature_details = {
            'Currency': {0: 'AUD', 1: 'CAD', 2: 'EUR', 3: 'GBP', 4: 'USD'}[int(features[6])],
        }

        logging.info("Processed delay prediction request.")
        return jsonify({'prediction': "~" + str(
            round(prediction, 2)) + f" {feature_details['Currency']} predicted Raised Amount of this Campaign"}), 200
    except Exception as e:
        logging.error(f"Error in delay prediction: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
