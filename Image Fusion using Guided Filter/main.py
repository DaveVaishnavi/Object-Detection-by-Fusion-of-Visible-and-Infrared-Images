import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

image1_path = 'infrared_image.png'
image2_path = 'visible_image.png'
kernel_size = (31, 31)
image_1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
image_2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
min_height = min(image_1.shape[0], image_2.shape[0])
min_width = min(image_1.shape[1], image_2.shape[1])
image_1 = image_1[:min_height, :min_width]
image_2 = image_2[:min_height, :min_width]
eps = 1e-6


# print("Image 1 dimensions:", image_1.shape)
# print("Image 2 dimensions:", image_2.shape)


def guided_filter(p, I, r=31, eps=1e-6):
    I_mean = gaussian_filter(I, sigma=r)
    p_mean = gaussian_filter(p, sigma=r)
    Ip_mean = gaussian_filter(I * p, sigma=r)
    I2_mean = gaussian_filter(I ** 2, sigma=r)
    cov_Ip = Ip_mean - I_mean * p_mean
    var_I = I2_mean - I_mean ** 2
    a = cov_Ip / (var_I + eps)
    b = p_mean - a * I_mean
    a_mean = gaussian_filter(a, sigma=r)
    b_mean = gaussian_filter(b, sigma=r)
    q = a_mean * I + b_mean
    return q


def two_scale_decomposition(image):
    base_layer = cv2.filter2D(image.astype(float), -1, np.ones(kernel_size) / np.prod(kernel_size),
                              borderType=cv2.BORDER_REFLECT)
    detail_layer = image - base_layer
    return base_layer, detail_layer


base_layer_1, detail_layer_1 = two_scale_decomposition(image_1)
base_layer_2, detail_layer_2 = two_scale_decomposition(image_2)
# print("Base Layer 1 dimensions:", base_layer_1.shape)
# print("Base Layer 2 dimensions:", base_layer_2.shape)
# print("Detail Layer 1 dimensions:", detail_layer_1.shape)
# print("Detail Layer 2 dimensions:", detail_layer_2.shape)


def construct_weight_map(detail_layer):
    detail_squared = detail_layer * detail_layer
    mean_detail_squared = cv2.filter2D(detail_squared, -1, np.ones(kernel_size) / np.prod(kernel_size),
                                       borderType=cv2.BORDER_REFLECT)

    detail_mean = cv2.filter2D(detail_layer, -1, np.ones(kernel_size) / np.prod(kernel_size),
                               borderType=cv2.BORDER_REFLECT)

    local_variance = mean_detail_squared - detail_mean * detail_mean
    local_variance_guided = guided_filter(detail_mean, local_variance)

    saliency_map = np.sqrt(np.maximum(local_variance_guided, 0))
    return saliency_map


weight_map_1 = construct_weight_map(detail_layer_1)
weight_map_2 = construct_weight_map(detail_layer_2)
# print("Weight Map 1 dimensions:", weight_map_1.shape)
# print("Weight Map 2 dimensions:", weight_map_2.shape)
fused_weight_map = np.maximum(weight_map_1, weight_map_2)


def two_scale_reconstruction(base_layer, detail_layer, weight_map):
    guided_detail = guided_filter(base_layer, detail_layer)
    reconstructed_detail = guided_detail * weight_map
    reconstructed_image = base_layer + reconstructed_detail
    return reconstructed_image


fused_image_1 = two_scale_reconstruction(base_layer_1, detail_layer_1, fused_weight_map)
fused_image_2 = two_scale_reconstruction(base_layer_2, detail_layer_2, fused_weight_map)
final_fused_image = 0.5 * (fused_image_1 + fused_image_2)
final_fused_image_normalized = (final_fused_image * 255).astype(np.uint8)
cv2.imshow('Final Fused Image', final_fused_image_normalized)
cv2.waitKey(0)
cv2.destroyAllWindows()
