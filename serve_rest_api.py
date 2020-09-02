import os
import urllib.request
from flask import Flask, jsonify, request
import cv2
from flask_cors import CORS
import CNNModel1
import torch

app = Flask(__name__)
cors = CORS(app, resources={r"/todo/api/v1.0/*": {"origins": "*"}})

img_width, img_height = 224, 224
inv_mapping = {0: 'normal', 1: 'COVID-19'}

# Specify the pytorch model path
PATH = "models/model_scratch.pt"

def load_image_firebase(image_path):
    print('load_image_firebase')
    print(image_path)
    file_name = format_file_name(image_path)
    print(file_name)
    urllib.request.urlretrieve(image_path, file_name)
    img = cv2.imread(file_name)
    print (img.shape)
    img = cv2.resize(img, (img_width, img_height))
    img = img.astype('float32') / 255.0

    return img

def format_file_name(image_path):
    url_split = image_path.split("/")
    file_name = url_split[-1]
    file_name = file_name.replace("?","")
    file_name = file_name.replace("=","")
    file_name = file_name.replace("&","")
    file_name = file_name.replace("%2F","")
    file_name = file_name + '.jpg'

    return file_name

def load_pytorch_model():
    global model
    model = CNNModel1.Net()
    model.load_state_dict(torch.load(PATH))
    model.eval()

@app.route('/todo/api/v1.0/covid19', methods=['POST'])
def get_covid19_prediagnosis():
    print('get_covid19_prediagnosis')
    content = request.json
    print(content)
    imagenUrl = content['imagenUrl']
    print(imagenUrl)
    img_tensor = load_image_firebase(imagenUrl)
    input_data = torch.Tensor(img_tensor)
    print ('image loaded')
    result = model(input_data)
    print ('before final return')
    file_name = format_file_name(imagenUrl)
    os.remove(file_name)
    print('Prediction: {}'.format(inv_mapping[result.argmax(axis=1)[0]]))

    return jsonify({'result': result.tolist()})

if __name__ == '__main__':
    load_pytorch_model()
    app.run(host="0.0.0.0", debug=True)
