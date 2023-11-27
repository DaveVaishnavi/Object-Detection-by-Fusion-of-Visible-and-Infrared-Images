import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
kernel_size = (31, 31)


def two_scale_decomposition(image):
    base_layer = cv2.boxFilter(image, -1, kernel_size, normalize=True)
    detail_layer = image - base_layer
    return base_layer, detail_layer


def guided_filter(p, i, r=31, eps=1e-6):
    I_mean = gaussian_filter(i, sigma=r)
    p_mean = gaussian_filter(p, sigma=r)
    Ip_mean = gaussian_filter(i * p, sigma=r)
    I2_mean = gaussian_filter(i ** 2, sigma=r)
    cov_Ip = Ip_mean - I_mean * p_mean
    var_I = I2_mean - I_mean ** 2
    a = cov_Ip / (var_I + eps)
    b = p_mean - a * I_mean
    a_mean = gaussian_filter(a, sigma=r)
    b_mean = gaussian_filter(b, sigma=r)
    q = a_mean * i + b_mean
    return q


def saliency_measure(image):
    LaplacianFilter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    Hn = cv2.filter2D(image, cv2.CV_32F, LaplacianFilter)
    local_mean = cv2.blur(Hn, (3, 3))
    Sn = np.abs(local_mean)
    rg = 5
    sigma_g = 5
    g = cv2.getGaussianKernel(2 * rg + 1, sigma_g)
    saliency_map = cv2.filter2D(Sn, cv2.CV_32F, g * g.T)
    return saliency_map


def examined_weight_maps(w_map_1, w_map_2):
    rows, columns = w_map_1.shape
    pa1 = np.zeros((rows, columns))
    pa2 = np.zeros((rows, columns))

    for i in range(rows):
        for j in range(columns):
            if w_map_1[i, j] > w_map_2[i, j]:
                pa1[i, j] = 1
            elif w_map_1[i, j] < w_map_2[i, j]:
                pa2[i, j] = 1
            else:
                pa1[i, j] = 1
                pa2[i, j] = 1
    return pa1, pa2


# Read the visible and infrared images
visible_image = cv2.imread('visible_image.png')
infrared_image = cv2.imread('infrared_image.png')

template_image = cv2.imread('infrared_image.png', cv2.IMREAD_GRAYSCALE)
input_image = cv2.imread('visible_image.png', cv2.IMREAD_GRAYSCALE)

# Perform the alignment and fusion process
max_iterations = 100
epsilon = 1e-6
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iterations, epsilon)
transform_matrix = cv2.findTransformECC(template_image, input_image, np.eye(2, 3, dtype=np.float32),
                                            cv2.MOTION_EUCLIDEAN, criteria)[1]
print("Transformation Matrix:")
print(transform_matrix)

warped_ir = cv2.warpAffine(input_image, transform_matrix, (visible_image.shape[1], visible_image.shape[0]))
# cv2.imshow('Warped Image', warped_ir)

# Saliency and weight maps
Sn1 = saliency_measure(input_image)
Sn2 = saliency_measure(template_image)
cv2.imshow("Sn1", Sn1)
cv2.imshow("Sn2", Sn2)

p1, p2 = examined_weight_maps(Sn1, Sn2)
p1_colored = p1[:, :, np.newaxis]
p2_colored = p2[:, :, np.newaxis]
cv2.imshow("P1", p1)
cv2.imshow("P2", p2)

# Two-scale decomposition
base_layer_1, detail_layer_1 = two_scale_decomposition(visible_image)
base_layer_2, detail_layer_2 = two_scale_decomposition(infrared_image)
cv2.imshow("B1", base_layer_1)
cv2.imshow("D1", detail_layer_1)
cv2.imshow("B2", base_layer_2)
cv2.imshow("D2", detail_layer_2)

wb1 = guided_filter(p1_colored, base_layer_1)
wb2 = guided_filter(p2_colored, base_layer_2)
wd1 = guided_filter(p1_colored, detail_layer_1)
wd2 = guided_filter(p2_colored, detail_layer_2)
cv2.imshow("WB1", wb1)
cv2.imshow("WB2", wb2)
cv2.imshow("WD1", wd1)
cv2.imshow("WD2", wd2)

# Fusion
fused_base_layer = (wb1 * base_layer_1) + (wb2 * base_layer_2)
fused_detail_layer = (wd1 * detail_layer_1) + (wd2 * detail_layer_2)
fused_image = fused_base_layer + fused_detail_layer
cv2.imshow("Fused Image", fused_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
