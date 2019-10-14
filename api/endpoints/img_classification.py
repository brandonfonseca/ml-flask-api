from flask import Blueprint, jsonify, request
from api.models.CIFAR10.lenet import LeNet
from api.exceptions.UnprocessableEntity import UnprocessableEntity
import torch
import requests
from PIL import Image
from torchvision import transforms
import json

# configures blueprint for image classification endpoint
img_classification_api = Blueprint('img_classification_api', __name__)

# potential classes for prediction
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

device = torch.device('cpu') # load the model on CPU
model = LeNet()  # initialize model architecture (imported at the beginning)

# load the trained model parameters
model.load_state_dict(torch.load('api/models/CIFAR10/cifar_parameters.pt', map_location=device))

model.eval()  # set the model to evaluation mode so dropouts and gradient calculations are disabled

# Apply transformations to the input image so it can be interpreted by the model
transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


@img_classification_api.route('/img_classification', methods=['POST'])  # configuring the image classification endpoint
def img_classification():
    try:  # attempt to load the input payload
        payload = json.loads(request.data)
    except json.decoder.JSONDecodeError:
        raise UnprocessableEntity('Unable to read JSON data. Please ensure that your data is correctly formatted.', status_code=422)
    ret_obj = {}
    prediction, confidence = get_pred(payload)  # get the model prediction
    if not prediction:
        ret_obj["Message"] = "Please ensure a valid image url is passed in the payload."

    # consider the model prediction if it has greater than 50% confidence in its prediction
    elif float(confidence) > 50:
        ret_obj["prediction"] = prediction
        ret_obj["confidence"] = confidence + " %"

    else:
        ret_obj["message"] = "Sorry, I am unsure what this image is. Please make sure the image falls into 1 of the following 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck"

    return jsonify(ret_obj) # return result to user


def get_pred(payload):  # get model prediction by applying transformations, querying the model, and parsing the output
    img_url = payload["img_url"]
    if not img_url:
        return False, False
    try:
        response = requests.get(img_url, stream = True)
        img = Image.open(response.raw)
        img = transform(img)
        image = img.unsqueeze(0)
        output = model(image)
        _, pred = torch.max(output, 1)  # gets the prediction of the model
        prediction = classes[pred.item()]
        confidence = output[0][pred.item()].tolist() * 100
        return prediction, str(confidence)
    except:
        return False, False
