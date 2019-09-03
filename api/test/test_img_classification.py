import unittest
import json
from flask import Flask
from api.endpoints.img_classification import img_classification_api

app = Flask(__name__)
app.register_blueprint(img_classification_api)


class ImgClassificationTests(unittest.TestCase):

    tester = None

    def __init__(self, *args, **kwargs):
        super(ImgClassificationTests, self).__init__(*args, **kwargs)
        global tester
        tester = app.test_client()

    def test_img_classification(self):
        response = tester.post(
            '/img_classification',
            data=json.dumps({"img_url": "https://www.medicalnewstoday.com/content//images/articles/322/322868/golden-retriever-puppy.jpg"}),
            content_type='application/json'
        )

        data = json.loads(response.get_data(as_text=True))

        self.assertEqual(response.status_code, 200)
        self.assertEqual("dog", data["prediction"])


if __name__ == '__main__':
    unittest.main()
