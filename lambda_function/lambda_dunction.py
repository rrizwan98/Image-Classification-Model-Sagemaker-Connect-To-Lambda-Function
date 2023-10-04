import numpy as np
import requests
from PIL import Image
import json
import os
import io
from io import BytesIO
import boto3
import base64

all_labels = {0: 'daisy', 1: 'dandelion', 2: 'rose', 3: 'sunflower', 4: 'tulip'}

def lambda_handler(event, context):
    # DOWNLOAD IMAGE USING URL & SAVE INTO TMP FOLDER
    url_pet = 'https://thumbs.dreamstime.com/z/pink-tulip-bloom-red-beautiful-tulips-field-spring-time-sunlight-floral-background-garden-scene-holland-netherlands-europe-70951485.jpg?w=992'  # pet image
    response = requests.get(url_pet)
    img1 = Image.open(BytesIO(response.content))
    img1.save('/tmp/test.jpg')
    print(os.listdir('/tmp/'))
    
    file_name = '/tmp/test.jpg'
    print("***load image successfully***")
    
    # ACCESS TO THE SAGEMAKER ENDPOINT
    ENDPOINT_NAME = os.environ['ENDPOINT'] # SAGEMAKER ENDPOINT connects to the environment variable
    runtime = boto3.Session().client(service_name='runtime.sagemaker')
    print("***Read End-Point successfully***")
    
    # READ IMAGE  AND IMAGE PROCESSING
    test_image = Image.open(file_name)
    test_image = test_image.resize((300, 300))
    test_image = np.asarray(test_image)/255.0
    test_image=np.expand_dims(test_image, axis=0)
    
    print("shape", test_image.shape)
    
    # PREDICTION
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                      ContentType='application/json',
                                      Body=json.dumps(test_image.tolist()))
    print("***predictions successfully***")
    
    # RETURN THE PREDECTED RESULT
    result = response['Body'].read()
    result = json.loads(result)
    prediction = np.array(result['predictions'])
    print("result:",prediction)
    mp = np.max(prediction[0], axis=-1)
    
    labels = all_labels
    
    print("\n\nMaximum Probability: ", mp)
    predicted_class = labels[np.argmax(prediction[0], axis=-1)]
    print("Classified:", predicted_class, "\n\n")
    

    if mp >= 0.70:
        prediction = predicted_class
        return prediction
    else:
        prediction = 'please upload a valid image'
        return prediction
        

    
    