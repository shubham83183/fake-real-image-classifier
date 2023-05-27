import numpy as np
from radial_profile import azimuthalAverage
from scipy.interpolate import griddata


def unrolled_xy(magnitude):
    xy = magnitude.flatten()
    xy_norm = xy / (np.max(xy))
    return xy_norm


def average_xy(magnitude):
    x_average = np.mean(magnitude, axis=0)
    y_average = np.mean(magnitude, axis=1)
    average = np.append(x_average, y_average)
    # average /= np.max(average)
    return average


def azimuthal_avg(magnitude):
    n = 100
    pow_spectrum = azimuthalAverage(magnitude, center=None)
    pow_spectrum = pow_spectrum / (np.max(pow_spectrum))
    points = np.linspace(0, n, num=pow_spectrum.size)
    xi = np.linspace(0, n, num=n)
    interpolated = griddata(points, pow_spectrum, xi, method='cubic')
    interpolated /= interpolated[0]
    processed_data = interpolated
    return processed_data


def preprocess(data, method="unrolled_xy"):
    real_image = []
    bilinear = []
    bicubic = []
    pixel = []
    for i in range(len(data)):
        image, label = data[i]
        image = np.squeeze(image)
        fft = np.fft.fft2(image)
        fft_shift = np.fft.fftshift(fft)
        magnitude = 20 * np.log(np.abs(fft_shift + 1))
        if method == "unrolled_xy":
            processed_data = unrolled_xy(magnitude)
        elif method == "average_xy":
            processed_data = average_xy(magnitude)
        elif method == "azimuthal_avg":
            processed_data = azimuthal_avg(magnitude)

        #  {'SNGAN_bicubic': 0, 'SNGAN_bilinear': 1,
        #  'SNGAN_pixelshuffle': 2, 'imagewoof': 3}
        if label == 3:
            real_image.append(processed_data)
        elif label == 0:
            bicubic.append(processed_data)
        elif label == 1:
            bilinear.append(processed_data)
        elif label == 2:
            pixel.append(processed_data)
    # Converting to numpy array from list
    real_image = np.array(real_image)
    bicubic = np.array(bicubic)
    bilinear = np.array(bilinear)
    pixel = np.array(pixel)
    # appending real and fake image
    real_bicubic = np.append(real_image, bicubic, axis=0)
    real_bilinear = np.append(real_image, bilinear, axis=0)
    real_pixel = np.append(real_image, pixel, axis=0)
    #  appending labels corresponding to real(0) and fake (1) image
    label_bicubic = np.append(np.zeros(real_image.shape[0]), np.ones(bicubic.shape[0]))
    label_bilinear = np.append(np.zeros(real_image.shape[0]), np.ones(bilinear.shape[0]))
    label_pixel = np.append(np.zeros(real_image.shape[0]), np.ones(pixel.shape[0]))

    bicubic_data = (real_bicubic, label_bicubic)
    bilinear_data = (real_bilinear, label_bilinear)
    pixelshuffle_data = (real_pixel, label_pixel)

    return bicubic_data, bilinear_data, pixelshuffle_data
