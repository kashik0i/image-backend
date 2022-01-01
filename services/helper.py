import io
import os
import uuid
from base64 import encodebytes
from pprint import pprint

import cv2
import numpy as np
from PIL import Image


def make_unique_dir() -> str:
    directory = os.path.join(os.getcwd(), '../uploads', str(uuid.uuid4().hex))
    if not os.path.exists(directory):
        os.makedirs(directory)
    print(directory)
    return directory


def concat_images(im1, im2, out):
    img1 = cv2.imread(im1)
    img2 = cv2.imread(im2)
    vis = np.concatenate((img1, img2))
    cv2.imwrite(out, vis)


def get_input_output_path(input_image):
    uploads_dir = make_unique_dir()
    extension = os.path.splitext(input_image.filename)[1]
    if extension == '':
        extension = '.png'
    filename = str(uuid.uuid4().hex + extension)
    input_file_path = os.path.join(uploads_dir, filename)
    input_image.save(input_file_path)
    out_file_path = os.path.join(uploads_dir, "out-" + filename)
    pprint({'input_file_path': input_file_path, 'out_file_path': out_file_path, 'uploads_dir': uploads_dir,
            'extension': extension})
    return input_file_path, out_file_path, uploads_dir


def process_image(image):
    image = cv2.imread(image)
    image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
    return image


def get_response_image(image_path):
    pil_img = Image.open(image_path, mode='r')  # reads the PIL image
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='PNG')  # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii')  # encode as base64
    return encoded_img
