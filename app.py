import base64
import io
import os
import shutil
import traceback
import uuid
import json
from pprint import pprint

import cv2
from flask import Flask, send_file, render_template
from flask import request, Response
from flask_cors import CORS, cross_origin
from matplotlib import pyplot as plt
from numpy import ndarray
from werkzeug.utils import secure_filename
from services import fourier, convolution, noise, interpolation
from services.helper import make_unique_dir, concat_images, get_response_image, get_input_output_path

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'


# @app.route("/api/1")
# def print_api_routes():
#     return "div<p>Hello, World!</p>"
# from pathlib import Path
# Path("").mkdir(parents=True, exist_ok=True)

@app.route('/api/upload', methods=['POST'])
@cross_origin()
def upload_file():
    print(f"image{request.files}")
    input_image = request.files['image']
    input_image.save(f"uploads/{secure_filename(input_image.filename)}")
    return {
        "state": "success"
    }


@app.route('/api/convolution', methods=['POST'])
@cross_origin()
def convolution_route():
    # content_type = request.headers.get('Content-Type')
    pprint(request.files)
    input_image = request.files['input_image']
    kernel = request.form['kernel']
    kernel_name = request.form['kernel_name']
    kernel = json.loads(kernel)
    for i in range(0, 9):
        kernel[i] = float(kernel[i])
    try:
        input_file_path, out_file_path, uploads_dir = get_input_output_path(input_image)

        convolution.main(input_file_path, out_file_path, kernel, kernel_name)
    except Exception as e:
        traceback.print_exc()
        error = e.__str__()
        return {
            "status": "bad",
            "kernel": kernel,
            "error": error,
        }
    output = get_response_image(out_file_path)
    # shutil.rmtree(uploads_dir)
    return {
        "status": "good",
        "output_image": output,
        "kernel": kernel,
    }


@app.route('/api/interpolation', methods=['POST'])
@cross_origin()
def interpolation_route():
    pprint(request.files)
    input_image = request.files['input_image']
    width = int(request.form['width'])
    height = int(request.form['height'])
    try:
        input_file_path, out_file_path, uploads_dir = get_input_output_path(input_image)

        interpolation.main(input_file_path, out_file_path, width, height)
    except Exception as e:
        traceback.print_exc()
        error = e.__str__()
        return {
            "status": "bad",
            "error": error,
        }
    output = get_response_image(out_file_path)
    # shutil.rmtree(uploads_dir)
    return {
        "status": "good",
        "output_image": output,
    }


@app.route('/api/noise', methods=['POST'])
@cross_origin()
def noise_route():
    pprint(request.files)
    pprint(request.form)
    input_image = request.files['input_image']
    noise_type = request.form['noise']
    print(input_image.__sizeof__())
    try:
        input_file_path, out_file_path, uploads_dir = get_input_output_path(input_image)
        noise.main(input_file_path, out_file_path, noise_type)
    except Exception as e:
        traceback.print_exc()
        error = e.__str__()
        return {
            "status": "bad",
            "error": error,
        }
    output = get_response_image(out_file_path)
    # shutil.rmtree(uploads_dir)
    return {
        "status": "good",
        "output_image": output,
    }


@app.route('/api/fourier', methods=['POST'])
@cross_origin()
def fourier_route():
    # content_type = request.headers.get('Content-Type')
    print(request.files)
    input_image = request.files['input_image']
    try:
        input_file_path, out_file_path, uploads_dir = get_input_output_path(input_image)
        fourier.main(input_file_path, out_file_path)
    except Exception as e:
        traceback.print_exc()
        error = e.__str__()
        return {
            "status": "bad",
            "error": error,
        }
    output = get_response_image(out_file_path)
    # shutil.rmtree(uploads_dir)
    return {
        "status": "good",
        "output_image": output,
    }

# if __name__ == '__main__':
#     app.run('0.0.0.0', 50014, debug=True)
