import numpy as np
import cv2


def check_if_monochrome(image):
    if len(image.shape) == 2:
        return True
    if len(image.shape) == 3 and np.array_equal(image[:, :, 0], image[:, :, 1]) and np.array_equal(
            image[:, :, 0], image[:, :, 2]):
        return True
    return False


def calculate_histogram(image):
    histogram = np.zeros(256)
    for pixel_value in range(256):
        histogram[pixel_value] = np.sum(image == pixel_value)
    return histogram


def calculate_lut_arrays(image, is_monochrome):
    lut_arrays = {}
    if is_monochrome:
        lut_arrays['Intensity'] = calculate_lut_array(image)
    else:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lut_arrays['Intensity (weighted)'] = calculate_lut_array(gray_image)
        b, g, r = cv2.split(image)
        unweighted_image = np.round((r.astype(np.uint32) + g.astype(np.uint32) + b.astype(np.uint32)) / 3).astype(
            np.uint8)
        lut_arrays['Intensity (unweighted)'] = calculate_lut_array(unweighted_image)
        for i, color in enumerate(['R', 'G', 'B']):
            lut_arrays[color] = calculate_lut_array(image[:, :, 2 - i])

    return lut_arrays


def calculate_lut_array(channel):
    lut_array = np.zeros(256, dtype=np.uint32)
    height, width = channel.shape[:2]
    for i in range(height):
        for j in range(width):
            pixel_value = channel[i, j]
            lut_array[pixel_value] += 1
    return lut_array


def update_scale(scale, value):
    current_value = scale.get()
    new_value = max(scale['from'], min(scale['to'], current_value + value))
    scale.set(new_value)