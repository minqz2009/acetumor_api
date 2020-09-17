from flask import render_template, request, redirect, jsonify, Response, Blueprint
from PIL import Image
import os
from io import BytesIO
import base64
from acetumor.utils.predict import predict

api = Blueprint('api', __name__) 

@api.route("/upload-img", methods=['POST'])
def upload_img():
    try:
        js = request.get_json()
        position = js['position']
        img_str = js['img']
        img_str = img_str.encode("utf-8")
        img_str = base64.b64decode(img_str)
        img_str = BytesIO(img_str)
        reply, label = predict(position, Image.open(img_str).convert("RGB"))

        return reply, 200
    except Exception as e:
        return jsonify(status="error", details=str(e)), 400 