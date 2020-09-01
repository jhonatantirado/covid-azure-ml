import os
import urllib.request
import cloudinary.uploader
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from keras.preprocessing import image
import cv2
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app, resources={r"/todo/api/v1.0/*": {"origins": "*"}})

cloudinary.config(
    cloud_name="ds04o8pmi",
    api_key="618563954112198",
    api_secret="17I684A-O6lVnHh0fRQ9IPje1zQ"
)

img_width, img_height = 224, 224
inv_mapping = {0: 'normal', 1: 'pneumonia', 2: 'COVID-19'}
weightspath = 'model'
metaname = 'model.meta'
ckptname = 'model-1697'
image_tensor_name = 'input_1:0'
pred_tensor_name = 'dense_3/Softmax:0'

sess = None
image_tensor = None
pred_tensor = None

def load_image(image_path):
    url = cloudinary.utils.cloudinary_url(image_path)
    urllib.request.urlretrieve(url[0], image_path)
    img = cv2.imread(image_path)
    img = cv2.resize(img, (img_width, img_height))
    img = img.astype('float32') / 255.0

    return img

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

def load_model():
    global sess
    global image_tensor
    global pred_tensor

    sess = tf.Session()
    tf.get_default_graph()
    saver = tf.train.import_meta_graph(os.path.join(weightspath, metaname))
    saver.restore(sess, os.path.join(weightspath, ckptname))
    graph = tf.get_default_graph()
    image_tensor = graph.get_tensor_by_name(image_tensor_name)
    pred_tensor = graph.get_tensor_by_name(pred_tensor_name)

def predict(x):
    pred = sess.run(pred_tensor, feed_dict={image_tensor: np.expand_dims(x, axis=0)})
    return pred

@app.route('/todo/api/v1.0/prediagnosis/<string:image_url>', methods=['GET'])
def get_prediagnosis(image_url):
    img_tensor = load_image(image_url)
    print ('image loaded')
    result = predict(img_tensor)
    print ('before final return')
    os.remove(image_url)
    print('Prediction: {}'.format(inv_mapping[result.argmax(axis=1)[0]]))

    return jsonify({'result': result.tolist()})

@app.route('/todo/api/v1.0/covid19', methods=['POST'])
def get_covid19_prediagnosis():
    print('get_covid19_prediagnosis')
    content = request.json
    print(content)
    imagenUrl = content['imagenUrl']
    print(imagenUrl)
    img_tensor = load_image_firebase(imagenUrl)
    print ('image loaded')
    result = predict(img_tensor)
    print ('before final return')
    file_name = format_file_name(imagenUrl)
    os.remove(file_name)
    print('Prediction: {}'.format(inv_mapping[result.argmax(axis=1)[0]]))

    return jsonify({'result': result.tolist()})

if __name__ == '__main__':
    load_model()
    app.run(host="0.0.0.0", debug=True)
