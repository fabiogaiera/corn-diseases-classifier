# Corn Diseases Classifier API  

Given an image as input, the API classifies common diseases from corn.  

## Model training  

We are going to train a model using the dataset https://www.tensorflow.org/datasets/catalog/plant_village  

**Execute with Jupyter Notebook**  

* Python notebook `Dependencies.ipynb`  
* Python notebook `Model Training.ipynb`  

## API deployment that serves the model  

**Create environment for TensorFlow Lite**   
python -m venv .venv  

**Activate the environment**  
source `.venv/bin/activate` (Linux & MacOS) or `.venv\Scripts\activate` (Windows)  

**Install some dependencies**  
* pip install --upgrade pip  
* pip install fastapi  
* pip install uvicorn  
* pip install gunicorn   
* pip install tflite-runtime  
* pip install pillow  
* pip install python-multipart     

**Create requirements.txt file**     
* pip freeze > requirements.txt  

**Start FastAPI application**  
* Once the environment is active, just execute `uvicorn main:app --reload`    

**REST API Calls**  

* GET http://localhost:8000/api/ping (Postman collection available in repository)  

* POST http://localhost:8000/api/predict (Postman collection available in repository)  

**Heroku commands for troubleshooting**  

* heroku login  
* heroku run bash -a corn-diseases-classifier  
* heroku logs --tail -a corn-diseases-classifier  