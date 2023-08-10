import cv2
import numpy as np

image_path_1 = 'visible_image.png'
image_path_2 = 'infrared_image.png'
image_1 = cv2.imread(image_path_1, cv2.IMREAD_GRAYSCALE)
image_2 = cv2.imread(image_path_2, cv2.IMREAD_GRAYSCALE)
kernel_size = (31, 31)


def two_scale_decomposition(image):
    base_layer = cv2.filter2D(image.astype(float), -1, np.ones(kernel_size) / np.prod(kernel_size),
                              borderType=cv2.BORDER_REFLECT)
    detail_layer = image - base_layer
    return base_layer, detail_layer


base_layer_1, detail_layer_1 = two_scale_decomposition(image_1)
base_layer_2, detail_layer_2 = two_scale_decomposition(image_2)
cv2.imshow('Base Layer 1', base_layer_1.astype(np.uint8))
cv2.imshow('Detail Layer 1', detail_layer_1.astype(np.uint8))
cv2.imshow('Base Layer 2', base_layer_2.astype(np.uint8))
cv2.imshow('Detail Layer 2', detail_layer_2.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
