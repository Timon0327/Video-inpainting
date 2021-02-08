'''
#  filename: metric.py
#  implementation of PSNR and SSIM
#  Likun Qin, 2021
'''
import numpy as np


def psnr(f, g):
    '''
    calculate PSNR(Peak-Signal-Noise-Radio)
    :param f: the image to be tested, ndarray, [height, width]
    :param g: the ground truth of the image, ndarray, [height, width]
    :return: psnr, float
             if f and g do not match in shape, return None
    '''
    # check shapes of inputs
    if f.shape != g.shape:
        print('error, input do not match in shape!')
        return None

    # mean
    f_mean = np.mean(f)
    g_mean = np.mean(g)

    # variance
    f_var = np.var(f)
    g_var = np.var(g)

    # covariance
    covariance = np.cov(f.flatten(), g.flatten())
    cov = covariance[0, 1]

    # mean squared error
    mse = f_var + g_var - 2*cov + (f_mean - g_mean)**2

    # PSNR
    result = 10 * np.log10(255 ** 2 / mse)

    return result


def ssim(f, g):
    '''
    calculate SSIM(Structural Similarity Index Measure)
    :param f: the image to be tested, ndarray, [height, width]
    :param g: the ground truth of the image, ndarray, [height, width]
    :return: ssim, float
             if f and g do not match in shape, return None
    '''
    # parameters
    c1 = 1
    c2 = 1
    c3 = 1

    # check shapes of inputs
    if f.shape != g.shape:
        print('error, input do not match in shape!')
        return None

    # mean
    f_mean = np.mean(f)
    g_mean = np.mean(g)

    # variance
    f_std = np.std(f)
    g_std = np.std(g)

    # covariance
    covariance = np.cov(f.flatten(), g.flatten())
    cov = covariance[0, 1]

    # luminance
    l = (2 * f_mean * g_mean + c1) / (f_mean**2 + g_mean**2 + c1)

    # contrast
    c = (2 * f_std * g_std + c1) / (f_std**2 + g_std**2 + c2)

    # structure
    s = (cov + c3) / (f_std * g_std + c3)

    return l * c * s


def test():
    a = np.array([[1, 1, 1],
                  [0, 0, 0],
                  [1, 1, 1]])
    b = np.array([[1, 1, 1],
                  [1, 1, 1],
                  [1, 1, 1]])
    res = psnr(a, b)
    print(res)


if __name__ == '__main__':
    test()