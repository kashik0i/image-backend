import cv2
import numpy as np


def dft_matrix(input_img):
    rows = input_img.shape[0]
    cols = input_img.shape[1]
    t = np.zeros((rows, cols), complex)
    output_img = np.zeros((rows, cols), complex)

    m = np.arange(rows)
    n = np.arange(cols)
    x = m.reshape((rows, 1))
    y = n.reshape((cols, 1))

    for row in range(0, rows):
        M1 = 1j * np.sin(-2 * np.pi * y * n / cols) + np.cos(-2 * np.pi * y * n / cols)
        t[row] = np.dot(M1, input_img[row])
    for col in range(0, cols):
        M2 = 1j * np.sin(-2 * np.pi * x * m / cols) + np.cos(-2 * np.pi * x * m / cols)
        output_img[:, col] = np.dot(M2, t[:, col])
    f_shift = np.fft.fftshift(output_img)
    f_abs = np.abs(f_shift) + 1  # lie between 1 and 1e6
    f_bounded = 20 * np.log(f_abs)
    f_img = 255 * f_bounded / np.max(f_bounded)
    f_img = f_img.astype(np.uint8)
    return f_img


def main(input_path, output_path):
    image = cv2.imread(input_path, 0)

    dftma_lena = dft_matrix(image)
    cv2.imwrite(output_path, dftma_lena)
    # return dftma_lena
