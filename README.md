Nama : Dimas Alvianto
Kelas : IF22A
NIM : 23422009


import cv2
import numpy as np
import matplotlib.pyplot as plt

def negative_image(image):
    return 255 - image

def log_transformation(image):
    image = np.where(image == 0, 1, image)  # Hindari log(0)
    c = 255 / np.log(1 + np.max(image))
    return np.array(c * np.log(1 + image), dtype=np.uint8)

def power_law_transformation(image, gamma=1.0):
    return np.array(255 * (image / 255) ** gamma, dtype=np.uint8)

def histogram_equalization(image):
    return cv2.equalizeHist(image)

def histogram_normalization(image):
    min_val, max_val = np.min(image), np.max(image)
    if min_val == max_val:  # Cegah pembagian dengan nol
        return image
    return ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)

def plot_images(original, processed, title):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(processed, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Load Image
image_path = 'igambar makalah.jpg'
gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if gray_image is None:
    print(f"Error: Gambar '{image_path}' tidak ditemukan atau gagal dimuat.")
else:
    # Apply Transformations
    neg_image = negative_image(gray_image)
    log_image = log_transformation(gray_image)
    power_image = power_law_transformation(gray_image, gamma=0.5)
    equalized_image = histogram_equalization(gray_image)
    norm_image = histogram_normalization(gray_image)

    # Plot Results
    plot_images(gray_image, neg_image, 'Negative Image')
    plot_images(gray_image, log_image, 'Log Transformation')
    plot_images(gray_image, power_image, 'Power Law Transformation')
    plot_images(gray_image, equalized_image, 'Histogram Equalization')
    plot_images(gray_image, norm_image, 'Histogram Normalization')

===================================================================================================================================================================================================
Penjelasan Kondisi Input dan Output:

1. Citra Negatif:
Input: Gambar grayscale.
Output: Warna terang menjadi gelap dan sebaliknya.

2. Transformasi Log:
Input: Gambar grayscale.
Output: Kontras tinggi pada intensitas rendah, cocok untuk gambar gelap.

3. Transformasi Power Law (Gamma Correction):
Input: Gambar grayscale.
Output: Meningkatkan kontras berdasarkan nilai gamma.

4. Histogram Equalization:
Input: Gambar grayscale.
Output: Distribusi intensitas lebih merata, meningkatkan detail pada area gelap/terang.

5. Histogram Normalization:
Input: Gambar grayscale.
Output: Rentang intensitas disesuaikan ke [0, 255] untuk meningkatkan kontras.

6. Konversi RGB ke HSI:
Input: Gambar berwarna.
Output: Gambar dalam format HSI, cocok untuk deteksi warna dan segmentasi.

Cara Menentukan Thresholding pada HSI:
Ekstrak Kanal H (Hue): Digunakan untuk segmentasi warna.
Gunakan Rentang Threshold: Misalnya, untuk mendeteksi warna merah, gunakan nilai H dalam rentang tertentu.
Gunakan Masking: Terapkan threshold untuk memisahkan objek dengan warna yang diinginkan.
