# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

import os
import torch
import torch.nn as nn
from torchvision import transforms
import json

from azureml.core.model import Model

#new imports
import urllib.request
from PIL import Image

model_file_name = "model_scratch.pt"
classes = ['normal', 'COVID-19']

def init():
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For multiple models, it points to the folder containing all deployed models (./azureml-models)
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), model_file_name)
    model = torch.load(model_path, map_location=lambda storage, loc: storage)#This works for a full model, not only for the weights
    model.eval()


def run(input_data):
    imagenUrl = json.loads(input_data)['imagenUrl']
    input_data = load_input_image(imagenUrl)
    # get prediction
    with torch.no_grad():
        output = model(input_data)
        softmax = nn.Softmax(dim=1)
        pred_probs = softmax(output).numpy()[0]
        index = torch.argmax(output, 1)

    result = {"label": classes[index], "probability": str(pred_probs[index])}
    return result

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
