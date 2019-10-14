from flask import Flask, jsonify
import logging
from .exceptions.UnprocessableEntity import UnprocessableEntity
from .endpoints.img_classification import img_classification_api
from .endpoints.sentiment import sentiment_api


app = Flask(__name__) # initialize the flask app/API

# connect the blueprints to the API
app.register_blueprint(img_classification_api)
app.register_blueprint(sentiment_api)

# endpoint that is used to return the payload from the UnprocessableEntity class
@app.errorhandler(UnprocessableEntity)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response

# Configuring logging for the API
if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)

# run the API
if __name__ == '__main__':
    app.run(host='0.0.0.0')
