from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flasgger import Swagger
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

# Load environment variables from .env file
# load_dotenv()

app = Flask(__name__)

# Configure Swagger
swagger_config = {
    "definitions": {"name":"Sparta 300"},
    'swagger': '2.0',
    'info': {
        'version': '0.0.1',
        'title': 'Campaign Prediction API',
        'description': """This API provides the prediction API which you can predict the Raised Amount""",
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
swagger = Swagger(app,config=swagger_config)

# Use an application context to access current_app
with app.app_context():
    print(swagger.get_apispecs(endpoint='apispec'))

# # Environment-based configuration for sensitive information
# API_URL = os.getenv("HUGGINGFACE_API_URL")
# API_KEY = os.getenv("HUGGINGFACE_API_KEY")
# HEADERS = {"Authorization": f"Bearer {API_KEY}"}

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
# model2 = joblib.load('Cost_Overrun_model.pkl')
# model3 = joblib.load('Injury_model.pkl')

# GPT Prompt Generation functions for different types of predictions
# def format_delay_prompt(prediction, feature_details):
#     """Generates a GPT prompt for delay prediction solutions"""
#     return (
#         f"A construction project with an estimated delay of {int(prediction)} days has been analyzed based on various factors. "
#         f"Key features include Estimated Area (Sq m): {feature_details['Estimated Area']}, "
#         f"Estimated Height: {feature_details['Estimated Height']}, Number of Workers: {feature_details['Number of Workers']}, "
#         f"Government Regulation Factor (%): {feature_details['Gov Regulation Factor']}, "
#         f"Site Condition Factor (%): {feature_details['Site Condition Factor']}, "
#         f"Bad Weather Factor: {feature_details['Bad Weather Factor']}, and Design Variation Factor: {feature_details['Design Variation Factor']}. "
#         f"Given this context, provide brief solutions (in points) (250-300 words) to prevent or minimize construction delays."
#     )

# def format_cost_prompt(prediction, feature_details):
#     """Generates a GPT prompt for cost overrun solutions"""
#     return (
#         f"A construction project with an estimated delay of {int(prediction)} days has been analyzed based on various factors. "
#         f"A construction project with a predicted cost overrun of {feature_details['Cost Overrun']} has been analyzed with "
#         f"the following significant factors: Inflation: {feature_details['Inflation']}, Structural Design Variation: "
#         f"{feature_details['Structural Design Variation']}, Cash Flow: {feature_details['Cash Flow']}, Resource Wastage: "
#         f"{feature_details['Resource Wastage']}, Good Coordination: {feature_details['Good Coordination']}, Contractor "
#         f"Experience: {feature_details['Contractor Experience']}, Equipment Breakdown: {feature_details['Equipment Breakdown']}, "
#         f"Budget: {feature_details['Budget']}, and Project Delay: {feature_details['Project Delay']}. Please provide solutions "
#         f"(300-400 words) to prevent or minimize this predicted cost overrun."
#     )

# def format_injury_prompt(prediction, feature_details):
#     """Generates a GPT prompt for injury prevention strategies"""
#     return (
#         f"A construction project with an estimated delay of {int(prediction)} days has been analyzed based on various factors. "
#         f"A construction project has a predicted injury risk categorized as {feature_details['Injury Risk']}. "
#         f"The contributing factors include Division: {feature_details['Division']}, Role: {feature_details['Role']}, Primary Cause: "
#         f"{feature_details['Primary Cause']}, Working Condition: {feature_details['Working Condition']}, Machine Condition: "
#         f"{feature_details['Machine Condition']}, Observation Type: {feature_details['Observation Type']}, and Incident Type: "
#         f"{feature_details['Incident Type']}. Provide a 300-400 word strategy to prevent or reduce workplace injuries for such a project."
#     )

# Async function to handle GPT API queries with retries
# async def query_gpt_model_async(prompt):
#     retries = 3
#     for attempt in range(retries):
#         try:
#             async with aiohttp.ClientSession() as session:
#                 async with session.post(API_URL, headers=HEADERS, json=prompt) as response:
#                     if response.status == 200:
#                         return await response.json()
#                     else:
#                         logging.warning(f"GPT API call failed: Status {response.status}")
#                         await asyncio.sleep(20 ** attempt)  # Exponential backoff
#         except Exception as e:
#             logging.error(f"GPT API call error: {e}")
#     return {'error': 'Failed after retries'}

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
            'Currency': {0:'AUD', 1:'CAD', 2:'EUR', 3:'GBP', 4:'USD'}[int(features[6])],
        }
        # prompt = format_delay_prompt(prediction, feature_details)
        # gpt_response = await query_gpt_model_async({"model": "meta-llama/Llama-3.2-3B-Instruct", "messages": [{"role": "user", "content": prompt}], "max_tokens": 1000, "temperature": 0.3})
        # solution = gpt_response.get('choices', [{'message': {'content': 'Solution not available'}}])[0]['message']['content']

        logging.info("Processed delay prediction request.")
        return jsonify({'prediction': "~"+str(str(round(prediction, 2))  +  f" {feature_details["Currency"]} predicted Raised Amount of this Campaign")}), 200
    except Exception as e:
        logging.error(f"Error in delay prediction: {e}")
        return jsonify({'error': str(e)}), 500

# Endpoint for cost overrun prediction
# @app.route('/cost_predict', methods=['POST'])
# @limiter.limit("100 per minute")
# async def prediction2():
#     """
#     Predicts cost overrun in construction and generates GPT solutions.
#     ---
#     parameters:
#       - name: features
#         in: body
#         schema:
#           type: array
#           items:
#             type: number
#         required: true
#         description: Features for cost overrun prediction.
#     responses:
#       200:
#         description: Cost overrun prediction and GPT solution
#       500:
#         description: Internal server error
#     """
#     try:
#         data = request.json
#         features = np.array(data['features'])
#         prediction = model2.predict([features])[0]
#
#         feature_details = {
#             'Inflation': {0: 'high', 1: 'low'}[int(features[0])],
#             'Structural Design Variation': {0: '1', 1: '2', 2: '3', 3: '4', 4: 'More than 4'}[int(features[1])],
#             'Cash Flow': {0: 'bad', 1: 'good'}[int(features[2])],
#             'Resource Wastage': {0: 'low', 1: 'medium'}[int(features[3])],
#             'Good Coordination': {0: 'no', 1: 'yes'}[int(features[4])],
#             'Contractor Experience': {0: 'Less than 2 years', 1: 'More than 2 years', 2: 'More than 4 years',3: 'More than 6 years', 4: 'More than 8 years', 5: 'More than 20 years'}[int(features[5])],
#             'Equipment Breakdown': {0: 'low', 1: 'medium'}[int(features[6])],
#             'Budget': {0: 'loose', 1: 'tight'}[int(features[7])],
#             'Project Delay': {0: 'less than 20%', 1: 'more than 20%', 2: 'more than 40%', 3: 'more than 60%',4: 'more than 80%'}[int(features[8])],
#             'Cost Overrun':{1: '~20% of the initial budget', 2: '~40% of the initial budget', 3: '~60% of the initial budget',4: '~80% of the initial budget', 5: '~99% of the initial budget'}[int(prediction)]
#         }
#         prompt = format_cost_prompt(prediction, feature_details)
#         gpt_response = await query_gpt_model_async({"model": "meta-llama/Llama-3.2-3B-Instruct", "messages": [{"role": "user", "content": prompt}], "max_tokens": 1000, "temperature": 0.3})
#         solution = gpt_response.get('choices', [{'message': {'content': 'Solution not available'}}])[0]['message']['content']
#
#         logging.info("Processed cost overrun prediction request.")
#         return jsonify({'prediction': feature_details["Cost Overrun"], 'solution': solution}), 200
#     except Exception as e:
#         logging.error(f"Error in cost overrun prediction: {e}")
#         return jsonify({'error': str(e)}), 500
#
# # Endpoint for injury risk prediction
# @app.route('/injury_predict', methods=['POST'])
# @limiter.limit("100 per minute")
# async def prediction3():
#     """
#     Predicts injury risk in construction and generates GPT solutions.
#     ---
#     parameters:
#       - name: features
#         in: body
#         schema:
#           type: array
#           items:
#             type: number
#         required: true
#         description: Features for injury risk prediction.
#     responses:
#       200:
#         description: Injury risk prediction and GPT solution
#       500:
#         description: Internal server error
#     """
#     try:
#         data = request.json
#         features = np.array(data['features'])
#         prediction = model3.predict([features])[0]
#
#         feature_details = {
#             'Division': {5: 'Metals', 0: 'Corporate Services', 3: 'Engineering & Project', 1: 'Others',4: 'Raw Materials', 2: 'Shared Services'}[int(features[0])],
#             'Role': {0: 'Contractor', 1: 'Employee'}[int(features[1])],
#             'Primary Cause':{4: 'Dashing/Collision', 5: 'Electricity/Fire/Energy', 6: 'Material Handling', 0: 'Medical Ailment',1: 'Process Incidents', 2: 'Slip/Trip/Fall', 3: 'Structural Integrity'}[int(features[2])],
#             'Working Condition': {0: 'Group Working', 1: 'Not Applicable', 2: 'Single Working'}[int(features[3])],
#             'Machine Condition': {0: 'Idle', 1: 'Not Applicable', 2: 'Working'}[int(features[4])],
#             'Observation Type': {0: 'Unsafe Act', 3: 'Unsafe Act & Unsafe Condition', 1: 'Unsafe Act by Other',2: 'Unsafe Condition'}[int(features[5])],
#             'Incident Type': {1: 'Behaviour', 0: 'Process'}[int(features[6])],
#             'Injury Risk': {8: 'High chances of fatal injury', 2: 'High chances of first aid injury',
#                             5: 'High chances of serious injury',
#                             6: 'Low chances of fatal injury', 0: 'Low chances of first aid injury',
#                             3: 'Low chances of serious injury',
#                             7: 'Medium chances of fatal injury', 1: 'Medium chances of first aid injury',
#                             4: 'Medium chances of serious injury'}[int(prediction)]
#         }
#         prompt = format_injury_prompt(prediction, feature_details)
#         gpt_response = await query_gpt_model_async({"model": "meta-llama/Llama-3.2-3B-Instruct", "messages": [{"role": "user", "content": prompt}], "max_tokens": 1000, "temperature": 0.3})
#         solution = gpt_response.get('choices', [{'message': {'content': 'Solution not available'}}])[0]['message']['content']
#
#         logging.info("Processed injury risk prediction request.")
#         return jsonify({'prediction': feature_details['Injury Risk'], 'solution': solution}), 200
#     except Exception as e:
#         logging.error(f"Error in injury risk prediction: {e}")
#         return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
