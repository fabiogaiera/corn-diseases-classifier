# Corn Diseases Classifier API

## The code works with TensorFlow 2.10.0

## Model training with dataset https://www.tensorflow.org/datasets/catalog/plant_village

1. Execute python notebook in notebook/Model Training.ipynb 

## API deployment that serves the model

1. python -m venv .venv

2. source `.venv/bin/activate (Linux & MacOS)` or `.venv\Scripts\activate (Windows)`

3. pip install -r requirements.txt

4. Create main.py file (main.py file already created in repository)

5. uvicorn main:app --reload

6. REST API Calls 

GET http://localhost:8000/api/ping (Postman collection available in repository)

POST http://localhost:8000/api/predict (Postman collection available in repository)


**Create environment for TensorFlow Lite**   
python -m venv .tflite-env  

**Activate the environment**   
source .tflite-env/bin/activate   

**Install some dependencies**  
pip install --upgrade pip  
pip install fastapi  
pip install uvicorn  
pip install gunicorn  
pip install tflite-runtime   
pip install pillow  
pip install python-multipart   

**Create requirements.txt file**   
pip freeze > requirements.txt  

**Start FastAPI application**  
Once the environment is active, just execute `uvicorn main:app --reload`  

**Heroku commands for troubleshooting**  

heroku login

heroku run bash -a scenes-classifier

heroku logs --tail -a scenes-classifier
