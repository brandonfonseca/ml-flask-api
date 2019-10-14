import unittest
import json
from flask import Flask
from api.endpoints.sentiment import sentiment_api

# Initializes a flask API instance for testing the image classification endpoint
app = Flask(__name__)
app.register_blueprint(sentiment_api)


class SentimentTests(unittest.TestCase):

    tester = None

    # Initialize the tester
    def __init__(self, *args, **kwargs):
        super(SentimentTests, self).__init__(*args, **kwargs)
        global tester
        tester = app.test_client()

    # Querying the sentiment analysis endpoint with a positive string
    def test_sentiment(self):
        response = tester.post(
            '/sentiment',
            data=json.dumps({"sentence": "You are an amazing person with a kind heart"}),
            content_type='application/json'
        )

        # Parsing the response from the API
        data = json.loads(response.get_data(as_text=True))

        # Ensuring that the API is working correctly and that the model correctly predicts that the string is positive
        self.assertEqual(response.status_code, 200)
        self.assertEqual("Positive", data["prediction"])


if __name__ == '__main__':
    unittest.main()
