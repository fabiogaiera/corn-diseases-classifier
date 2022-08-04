# Corn Diseases Classifier

## The code works with TensorFlow 2.8.0

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
