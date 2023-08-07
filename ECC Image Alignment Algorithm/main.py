import cv2
import numpy as np
template_image = cv2.imread('visible_image.png', cv2.IMREAD_GRAYSCALE)
input_image = cv2.imread('infrared_image.png', cv2.IMREAD_GRAYSCALE)
max_iterations = 100
epsilon = 1e-6
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iterations, epsilon)
transform_matrix = cv2.findTransformECC(template_image, input_image, np.eye(2, 3, dtype=np.float32), cv2.MOTION_EUCLIDEAN, criteria)[1]
print("Transformation Matrix:")
print(transform_matrix)
