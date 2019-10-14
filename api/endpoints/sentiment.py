from flask import Blueprint, jsonify, request
from fastai.text import *
from api.exceptions.UnprocessableEntity import UnprocessableEntity
import json
from nostril import nonsense

sentiment_api = Blueprint('sentiment_api', __name__)  # configures blueprint for sentiment endpoint
learner = load_learner('api/models/sentiment', 'sentiment_model.pkl')  # loads the sentiment model


@sentiment_api.route('/sentiment', methods=['POST'])  # configures endpoint and POST URL to reach endpoint
def sentiment():
    try:  # load payload
        payload = json.loads(request.data)
    except json.decoder.JSONDecodeError:
        raise UnprocessableEntity('Unable to read JSON data. Please ensure that your data is correctly formatted.', status_code=422)
    ret_obj = {}
    prediction, status = get_pred(payload)  # get prediction from model
    if status:  # sets up the json object to be returned to the user
        ret_obj["sentence"] = payload["sentence"]
        ret_obj["prediction"] = prediction
    else:
        ret_obj["message"] = prediction
    return jsonify(ret_obj)


def get_pred(payload):  # function to parse the prediction from the sentiment model
    sentence = payload["sentence"]
    if sentence:
        if is_meaningful(sentence):  # checks if the sentence is not nonsense
            return (learner.predict(sentence)[0]).__str__(), True
        else:
            return "This sentence is too short or meaningless. Please try again.", False
    return "Please ensure a valid sentence is passed in the payload and that the payload follows the required format", False


def is_meaningful(sentence):  # exploits external library to determine if a sentence is nonsense/jibberish
    if len(sentence.replace(" ", "")) < 6:
        return False
    try:
        if nonsense(sentence):
            return False
    except:
        return False
    return True
