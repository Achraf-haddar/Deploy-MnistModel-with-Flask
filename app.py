from flask import Flask, render_template, request
# For reshaping 
from scipy.misc import imsave, imread, imresize
import numpy as np 
import keras.models
import re
# syst going to help us do some system level operations
import sys
import os 
# Define the path of the model
sys.path.append(os.path.abspath('./model'))
from load import *
import base64

# Init Flask app
app = Flask(__name__)

# Declare global variables
global model, graph
#model, graph = init()  # init() function in the /mdoel/load.py
model = init()

"""
def convertImage(imgData1): 
    imgstr = re.search(b'base64,(.*)',imgData1).group(1) #print(imgstr) 
    with open('output.png','wb') as output: 
        output.write(base64.b64decode(imgstr))
"""
# Decode it from base64 into raw representation
def convertImage(imgData1):
    imgstr = re.search(b'base64,(.*)',imgData1).group(1)
    with open('output.png', 'wb') as output:
        output.write(base64.b64decode(imgstr))

# Routing
@app.route('/')
def index():
    # Load the page index.html
    return render_template('index.html')

@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    # Get the data from the image
    imgData = request.get_data()
    convertImage(imgData)
    x = imread('output.png', mode = 'L')
    # Invert make it easier to predict
    x = np.invert(x)
    # The size that the model expects
    x = imresize(x, (28, 28))
    # make it 4D tensor
    x = x.reshape(1, 28, 28, 1)
    """
    with graph.as_default():
        out = model.predict(x)
        response = np.array_str(np.argmax(out))
        return response
    """
    out = model.predict(x)
    print("=======================================")
    print(np.argmax(out))
    print("=======================================")
    response = str(np.argmax(out))
    return response

if __name__ == '__main__':
    # Specify the port
    port = int(os.environ.get('PORT', 5000))
    # Run the app locally
    app.run(host='0.0.0.0', port=port)
