from flask import Blueprint, jsonify, request
from fastai.text import *
from api.exceptions.UnprocessableEntity import UnprocessableEntity
import json
from nostril import nonsense

sentiment_api = Blueprint('sentiment_api', __name__)
learner = load_learner('api/models/sentiment', 'sentiment_model.pkl')

@sentiment_api.route('/sentiment', methods=['POST'])
def sentiment():
    try:
        payload = json.loads(request.data)
    except json.decoder.JSONDecodeError:
        raise UnprocessableEntity('Unable to read JSON data. Please ensure that your data is correctly formatted.', status_code=422)
    ret_obj = {}
    prediction, status = get_pred(payload)
    if status:
        ret_obj["sentence"] = payload["sentence"]
        ret_obj["prediction"] = prediction
    else:
        ret_obj["message"] = prediction
    return jsonify(ret_obj)


def get_pred(payload):
    sentence = payload["sentence"]
    if sentence:
        if is_meaningful(sentence):
            return (learner.predict(sentence)[0]).__str__(), True
        else:
            return "This sentence is too short or meaningless. Please try again.", False
    return "Please ensure a valid sentence is passed in the payload and that the payload follows the required format", False


def is_meaningful(sentence):
    if len(sentence.replace(" ", "")) < 6:
        return False
    try:
        if nonsense(sentence):
            return False
    except:
        return False
    return True
