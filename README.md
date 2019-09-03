# ML-Flask-API
Production ready API built using Flask inside of a Docker container. Gunicorn is used as the WSGI which allows the Flask app to be production ready.


# Installation Instructions

1. Sign up for a free Docker account and install Docker on your machine using the following link:
https://hub.docker.com/signup . After Docker is installed ensure that at least 4gb of ram is allocated to the containers using the Docker preferences pane. Then, start the Docker service.

2. Clone this repository to your local machine.
3. Navigate to the root directory of this project and run the following command in your terminal: `docker-compose build`. This will build the docker containers for the Flask API and MYSQL database. All necessary dependencies will be installed. The initial build will take a little while (~ 15 minutes)
4. After the build is completed run the following command in your terminal: `docker-compose up`. This will activate the docker containers and allow them to be queried.
5. Now that the docker containers are up you should now be able to query the endpoints.

# Endpoints

## `/img_classification` [POST]

This endpoint predicts whether an image falls into 1 of the 10 following categories (based on CIFAR10 dataset):

* Airplane
* Automobile
* Bird
* Cat
* Deer
* Dog
* Frog
* Horse
* Ship
* Truck

### JSON Request Body:

To obtain an image URL simply Google Image search your query, select the image you want classified, right click this image, then press "Copy Image Address"
```
{
    "img_url": "https://www.medicalnewstoday.com/content//images/articles/322/322868/golden-retriever-puppy.jpg"
}
```

### JSON Response:
```
{
    "Confidence": "85.75446605682373",
    "Prediction": "dog"
}
```

## `/sentiment` [POST]

This endpoint predicts the sentiment of a sentence. Currently, this model only supports classifying between a positive or negative sentence, therefore neutral sentences will likely be misclassified.

Please be sure to pass a sufficiently long sentence (greater than 8 characters) that is grammatically correct.

Please note that this model tends to perform better on sentences that are longer and more detailed. If a sentence is too short it is often hard for the model to make a meaningful prediction.

 

### JSON Request Body (example 1):

```
{
    "sentence": "You are an amazing person with a kind heart"
}
```

### JSON Response (example 1):
```
{
    "Prediction": "Positive",
    "Sentence": "You are an amazing person with a kind heart"
}
```

### JSON Request Body (example 2):

```
{
    "sentence": "I am unhappy with you lately, and you are not a good person"
}
```

### JSON Response (example 2):
```
{
    "Prediction": "Negative",
    "Sentence": "I am unhappy with you lately, and you are not a good person"
}
```

### JSON Request Body (example 3):

```
{
    "sentence": "efhufhfhfhofhf"
}
```

### JSON Response (example 3):
```
{
    "Message": "This sentence is too short or meaningless. Please try again."
}
```

## Model Architectures

### Image Classification Endpoint:

I built this model using the LeNet CNN architecture. It was built in vanilla PyTorch and achieved a validation accuracy of ~73% on the CIFAR 10 dataset. The accuracy of this endpoint could be increased if a different model architecture was used (especially if transfer learning was used).

If you would like to see the code implementation of how this model was built please refer to this ipynb notebook:
https://github.com/brandonfonseca/PyTorch-Course-2019/blob/master/Notebook%20%237%20-%20CNN%20(CIFAR10%20Classification).ipynb
 
### Sentiment Analysis Endpoint:

I built this model using the FastAI.text open source library on top of PyTorch. This model was trained/validated on the Stanford Sentiment Treebank dataset available on the following link: https://nlp.stanford.edu/sentiment/.
This model achieved a validation accuracy of ~93%.

If you would like to see the code implemenation of this model please refer to the following ipynb notebook:
https://github.com/brandonfonseca/sentiment-analysis-stanford/tree/master

## Database Support

You will notice that there is a docker container configured for a MYSQL database. As of right now, this database is not being used, however I have left it in the project so that future endpoints can use it if needed.

## Running Tests

This project has been equipped with automated unittests to ensure code quality before deploying to a production environment. 

Steps to run tests:

1. Ensure that the docker containers are running
2. Open a new terminal window and run the following command to shell into the ml-api container: ` docker exec -it ml-api /bin/bash
3. Run the following command to run the tests: `python3 run_tests.py`

If all of the tests successfully pass you should see the following output in the terminal: `OK`
