from flask import Blueprint, jsonify, request
from api.models.CIFAR10.lenet import LeNet
from api.exceptions.UnprocessableEntity import UnprocessableEntity
import torch
import requests
from PIL import Image
from torchvision import transforms
import json


img_classification_api = Blueprint('img_classification_api', __name__)
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
device = torch.device('cpu')
model = LeNet()
model.load_state_dict(torch.load('api/models/CIFAR10/cifar_parameters.pt', map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


@img_classification_api.route('/img_classification', methods=['POST'])
def img_classification():
    try:
        payload = json.loads(request.data)
    except json.decoder.JSONDecodeError:
        raise UnprocessableEntity('Unable to read JSON data. Please ensure that your data is correctly formatted.', status_code=422)
    ret_obj = {}
    prediction, confidence = get_pred(payload)
    if not prediction:
        ret_obj["Message"] = "Please ensure a valid image url is passed in the payload."

    elif float(confidence) > 50:
        ret_obj["Prediction"] = prediction
        ret_obj["Confidence"] = confidence + " %"

    else:
        ret_obj["Message"] = "Sorry, I am unsure what this image is. Please make sure the image falls into 1 of the following 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck"

    return jsonify(ret_obj)


def get_pred(payload):

    img_url = payload["img_url"]
    if not img_url:
        return False, False
    try:
        response = requests.get(img_url, stream = True)
        img = Image.open(response.raw)
        img = transform(img)
        image = img.unsqueeze(0)
        output = model(image)
        _, pred = torch.max(output, 1) # gets the prediction of the model
        prediction = classes[pred.item()]
        confidence = output[0][pred.item()].tolist() * 100
        return prediction, str(confidence)
    except:
        return False, False
