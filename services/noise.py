import random

import cv2
import numpy as np


def gasuss_noise(img):
    mean = 0
    var = 50
    sigma = var ** 0.5
    # gaussian = np.zeros(img.shape, np.float32)
    gaussian = np.random.normal(mean, sigma, img.shape)  # np.zeros((224, 224), np.float32)

    noisy_image = np.zeros(img.shape, np.float32)

    if len(img.shape) == 2:
        noisy_image = img + gaussian
    else:
        noisy_image[:, :, 0] = img[:, :, 0] + gaussian
        noisy_image[:, :, 1] = img[:, :, 1] + gaussian
        noisy_image[:, :, 2] = img[:, :, 2] + gaussian

    cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    noisy_image = noisy_image.astype(np.uint8)
    return noisy_image


def sp_noise(image, prob):
    """
    Add salt pepper noise
         PROB: Noise ratio
    """
    output = np.zeros(image.shape)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def main(input_path,output_path, noise):
    input_img = cv2.imread(input_path, 0)
    if noise == 'salt_and_pepper':
        noisyimage = sp_noise(input_img, 0.04)
    elif noise == 'gaussian':
        noisyimage = gasuss_noise(input_img)
    else:
        raise ValueError('Bad noise type')
    cv2.imwrite(output_path, noisyimage)
    print(noisyimage)
