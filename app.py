from flask import Flask, render_template, request
import time 
import cv2
import requests
import numpy as np

# Global Variables
_url = 'https://visionlabo-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/eddce1c1-f8dc-49f9-9dce-b411b533a1de/classify/iterations/Iteration1/url'
_key = '719c2acc1cc14e1989cd623db4e1cf57'
_maxNumRetries = 10

def predict_image(json, data, headers, params):

    """
    Helper function to process the request to Project Oxford

    Parameters:
    json: Used when processing images from its URL. See API Documentation
    data: Used when processing image read from disk. See API Documentation
    headers: Used to pass the key information and the data type request
    """

    retries = 0
    result = None

    while True:

        response = requests.request( 'post', _url, json = json, data = data, headers = headers, params = params )

        if response.status_code == 429: 

            print( "Message: %s" % ( response.json() ) )

            if retries <= _maxNumRetries: 
                time.sleep(1) 
                retries += 1
                continue
            else: 
                print( 'Error: failed after retrying!' )
                break

        elif response.status_code == 200 or response.status_code == 201:

            if 'content-length' in response.headers and int(response.headers['content-length']) == 0: 
                result = None 
            elif 'content-type' in response.headers and isinstance(response.headers['content-type'], str): 
                if 'application/json' in response.headers['content-type'].lower(): 
                    result = response.json() if response.content else None 
                elif 'image' in response.headers['content-type'].lower(): 
                    result = response.content
        else:
            print( "Error code: %d" % ( response.status_code ) )
            print( "Message: %s" % ( response.json() ) )

        break
        
    return [
        result["predictions"][0]["tagName"],
        result["predictions"][1]["tagName"],
        result["predictions"][2]["tagName"],
        result["predictions"][3]["tagName"],
        result["predictions"][4]["tagName"]
        ]

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def process_url():
    url = request.form['url']
    
    params = {'visualFeatures': 'Color,Categories'} 

    headers = dict()
    headers['Prediction-Key'] = _key
    headers['Content-Type'] = 'application/json' 

    json = {'url': url} 

    # Placeholder code to fetch image from URL
    response = requests.get(url)
    image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)

    # Placeholder code to process image and get predictions
    predictions = predict_image(json, None, headers, params)

    # Save the processed image temporarily
    cv2.imwrite('static/predicted_image.jpg', image)

    # Render the template with the image URL and predictions
    return render_template('index.html', image_url='/static/predicted_image.jpg', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
