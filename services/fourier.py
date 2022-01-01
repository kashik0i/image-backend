import cv2
import numpy as np


# # import cv2
# # import numpy as np
# # import math
# # import matplotlib.pyplot as plt
# #
# #
# # def dft_matrix(input_img):
# #     rows = input_img.shape[0]
# #     cols = input_img.shape[1]
# #     t = np.zeros((rows, cols), complex)
# #     output_img = np.zeros((rows, cols), complex)
# #     m = np.arange(rows)
# #     n = np.arange(cols)
# #     x = m.reshape((rows, 1))
# #     y = n.reshape((cols, 1))
# #     for row in range(0, rows):
# #         m1 = 1j * np.sin(-2 * np.pi * y * n / cols) + np.cos(-2 * np.pi * y * n / cols)
# #         t[row] = np.dot(m1, input_img[row])
# #     for col in range(0, cols):
# #         m2 = 1j * np.sin(-2 * np.pi * x * m / cols) + np.cos(-2 * np.pi * x * m / cols)
# #         output_img[:, col] = np.dot(m2, t[:, col])
# #     return output_img
# #
# #
# # def process_image(image):
# #     image = cv2.imread(image)
# #     image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
# #     return image
# #
# #
# # def main(input_path, output_path: str):
# #     print(input_path)
# #     input_img = process_image(input_path)
# #     # np.fft.rfft2(input_img)
# #     # input_img = cv2.imread(input_path, 0)
# #     output_img = dft_matrix(input_img)
# #     print(output_img.dtype)
# #     print(output_img.max())
# #     print('saving: ', output_path)
# #     cv2.imwrite(output_path, np.log(np.abs(output_img)))
# import cv2
# import numpy as np
# import math
# from PIL import Image
# import matplotlib.pyplot as plt
#
# from services.helper import process_image
#
#
# def run(input_path, output_path: str):
#     # f = Fourier()
#     # input_img = process_image(input_path)
#     # dftma_lena = f.dft_matrix(input_img)
#     # out_dftma = np.log(np.abs(dftma_lena))
#     # cv2.imwrite(output_path, out_dftma)
#     # print(out_dftma)
#     img=plt.imread(input_path).astype(float)
#     spectrum = np.fft.fftshift(np.fft.fft2(img))
#     plt.imsave(output_path,np.log(np.abs(spectrum)) )
#     img_back = np.fft.ifft2(np.fft.ifftshift(spectrum))
#
#
# class Fourier:
#     def dft_matrix(self, input_img):
#         rows = input_img.shape[0]
#         cols = input_img.shape[1]
#         t = np.zeros((rows, cols), complex)
#         output_img = np.zeros((rows, cols), complex)
#         m = np.arange(rows)
#         n = np.arange(cols)
#         x = m.reshape((rows, 1))
#         y = n.reshape((cols, 1))
#         for row in range(0, rows):
#             M1 = 1j * np.sin(-2 * np.pi * y * n / cols) + np.cos(-2 * np.pi * y * n / cols)
#             t[row] = np.dot(M1, input_img[row])
#         for col in range(0, cols):
#             M2 = 1j * np.sin(-2 * np.pi * x * m / cols) + np.cos(-2 * np.pi * x * m / cols)
#             output_img[:, col] = np.dot(M2, t[:, col])
#         return output_img


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
