import os
import urllib.request
from flask import Flask, jsonify, request
from flask_cors import CORS
import CNNModel1
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

app = Flask(__name__)
cors = CORS(app, resources={r"/todo/api/v1.0/*": {"origins": "*"}})

img_width, img_height = 224, 224

# Specify the pytorch model path
PATH = "models/model_scratch.pt"

classes = ['normal', 'pneumonia']

def load_input_image(image_path):
    file_name = format_file_name(image_path)
    urllib.request.urlretrieve(image_path, file_name)
    image = Image.open(file_name).convert('RGB')
    prediction_transform = transforms.Compose([transforms.Resize(size=(img_width, img_height)),
                                     transforms.ToTensor()])
    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = prediction_transform(image)[:3,:,:].unsqueeze(0)
    return image

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
    content = request.json
    imagenUrl = content['imageUrl']
    img_tensor = load_input_image(imagenUrl)
    input_data = img_tensor

    with torch.no_grad():
        output = model(input_data)
        softmax = nn.Softmax(dim=1)
        pred_probs = softmax(output).numpy()[0]
        index = torch.argmax(output, 1)

    result = {"label": classes[index], "probability": str(pred_probs[index])}

    file_name = format_file_name(imagenUrl)
    os.remove(file_name)

    return jsonify({'result': result})

if __name__ == '__main__':
    load_pytorch_model()
    app.run(host="0.0.0.0", debug=True)
