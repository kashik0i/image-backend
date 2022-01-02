import traceback

import numpy as np
import cv2


def main(input_path: str, output_path: str, kernel, kernel_name: str):
    if kernel_name == 'median':
        removed_noise = median_filter(input_path, 3)
        cv2.imwrite(output_path, removed_noise)
        # img = Image.fromarray(removed_noise)
    elif kernel_name == 'sobel':
        im = sobel(input_path)
        cv2.imwrite(output_path, im)
    elif kernel_name == 'robert':
        im = robert(input_path)
        print(type(im))
        # gray = cv2.cvtColor(cv2.UMat(im), cv2.COLOR_RGB2GRAY)
        cv2.imwrite(output_path, im)
    elif kernel_name=="gaussian_blur":
        kernel=gaussian_kernel()
        im = convolve_2d(input_path, kernel, padding=0)
        cv2.imwrite(output_path, im)
    else:
        kernel = np.reshape(kernel, (3, 3))
        im = convolve_2d(input_path, kernel, padding=0)
        cv2.imwrite(output_path, im)


def robert(image_path):
    roberts_cross_v = np.array([[0, 0, 0],
                                [0, 1, 0],
                                [0, 0, -1]])

    roberts_cross_h = np.array([[0, 0, 0],
                                [0, 0, 1],
                                [0, -1, 0]])

    vertical = convolve_2d(image_path, roberts_cross_v)
    horizontal = convolve_2d(image_path, roberts_cross_h)
    edged_img = np.sqrt(np.square(horizontal) + np.square(vertical))

    return edged_img


def sobel(input_path):
    Gx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
    Gy = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
    input_image = cv2.imread(input_path)
    grayscale_image = cv2.cvtColor(src=input_image, code=cv2.COLOR_BGR2GRAY)
    [rows, columns] = np.shape(grayscale_image)  # we need to know the shape of the input grayscale image
    sobel_filtered_image = np.zeros(shape=(rows, columns))  # initialization of the output image array (all elements
    # are 0)

    for i in range(rows - 2):
        for j in range(columns - 2):
            gx = np.sum(np.multiply(Gx, grayscale_image[i:i + 3, j:j + 3]))  # x direction
            gy = np.sum(np.multiply(Gy, grayscale_image[i:i + 3, j:j + 3]))  # y direction
            sobel_filtered_image[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)  # calculate the "hypotenuse"

    return sobel_filtered_image


def median_filter(input_path, filter_size):
    data = cv2.imread(input_path, 0)
    temp = []
    indexer = filter_size // 2
    data_final = np.zeros((len(data), len(data[0])))
    for i in range(len(data)):

        for j in range(len(data[0])):

            for z in range(filter_size):
                if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                    for c in range(filter_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(filter_size):
                            temp.append(data[i + z - indexer][j + k - indexer])

            temp.sort()
            data_final[i][j] = temp[len(temp) // 2]
            temp = []
    return data_final


def convolve_2d(input_path, kernel, padding=0):
    image = cv2.imread(input_path)
    image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int((xImgShape - xKernShape + 2 * padding) + 1)
    yOutput = int((yImgShape - yKernShape + 2 * padding) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding * 2, image.shape[1] + padding * 2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        print(imagePadded)
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        for x in range(image.shape[0]):
            # Go to next row once kernel is out of bounds
            if x > image.shape[0] - xKernShape:
                break
            try:
                output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
            except:
                break

    return output


def gaussian_kernel(l=5, sig=21): #5 21
    """\
    Gaussian Kernel Creator via given length and sigma
    """

    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))

    return kernel / np.sum(kernel)

# def convolve_2d(input_path, kernel, padding=0, strides=1):
#     image = cv2.imread(input_path)
#     image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
#     # Cross Correlation
#
#     kernel = np.flipud(np.fliplr(kernel))
#
#     # Gather Shapes of Kernel + Image + Padding
#     x_kern_shape = kernel.shape[0]
#     y_kern_shape = kernel.shape[1]
#     x_img_shape = image.shape[0]
#     y_img_shape = image.shape[1]
#
#     # Shape of Output Convolution
#     x_output = int(((x_img_shape - x_kern_shape + 2 * padding) / strides) + 1)
#     y_output = int(((y_img_shape - y_kern_shape + 2 * padding) / strides) + 1)
#     output = np.zeros((x_output, y_output))
#
#     # Apply Equal Padding to All Sides
#     if padding != 0:
#         image_padded = np.zeros((image.shape[0] + padding * 2, image.shape[1] + padding * 2))
#         image_padded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
#         print(image_padded)
#     else:
#         image_padded = image
#
#     # Iterate through image
#     for y in range(image.shape[1]):
#         # Exit Convolution
#         if y > image.shape[1] - y_kern_shape:
#             break
#         # Only Convolve if y has gone down by the specified Strides
#         if y % strides == 0:
#             for x in range(image.shape[0]):
#                 # Go to next row once kernel is out of bounds
#                 if x > image.shape[0] - x_kern_shape:
#                     break
#                 try:
#                     # Only Convolve if x has moved by the specified Strides
#                     if x % strides == 0:
#                         output[x, y] = (kernel * image_padded[x: x + x_kern_shape, y: y + y_kern_shape]).sum()
#                 except Exception:
#                     traceback.print_exc()
#                     break
#
#     return output
