import unittest
import json
from flask import Flask
from api.endpoints.img_classification import img_classification_api

# Initializes a flask API instance for testing the image classification endpoint
app = Flask(__name__)
app.register_blueprint(img_classification_api)


class ImgClassificationTests(unittest.TestCase):

    tester = None

    # Initialize the tester
    def __init__(self, *args, **kwargs):
        super(ImgClassificationTests, self).__init__(*args, **kwargs)
        global tester
        tester = app.test_client()

    # Querying the image classification endpoint with a picture of a dog
    def test_img_classification(self):
        response = tester.post(
            '/img_classification',
            data=json.dumps({"img_url": "https://s3.amazonaws.com/cdn-origin-etr.akc.org/wp-content/uploads/2017/11/12234558/Chinook-On-White-03.jpg"}),
            content_type='application/json'
        )

        # Parsing the response from the API
        data = json.loads(response.get_data(as_text=True))

        # Ensuring that the API is working correctly and that the model correctly predicts that the image is a dog
        self.assertEqual(response.status_code, 200)
        self.assertEqual("dog", data["prediction"])


if __name__ == '__main__':
    unittest.main()
