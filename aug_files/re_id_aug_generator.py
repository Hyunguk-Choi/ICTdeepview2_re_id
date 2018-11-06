from __future__ import print_function, division
import aug_files.imgaug.augmenters as iaa
from scipy import ndimage, misc
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
import os
from PIL import Image
import scipy.misc

def draw_per_augmenter_images(m_training_img):
    ''' 총 16개 의 버전으로 augmentation(원본포함) '''
    Naug = 2
    if np.amax(m_training_img) > 1:
        m_training_img = np.uint8(m_training_img)  # image 처리를 위해서 정수화

    nSample = m_training_img.shape[0]   # [ number of data, H, W, D]
    output_img = np.zeros((nSample * Naug, m_training_img[0].shape[0], m_training_img[0].shape[1], m_training_img[0].shape[2]))
    # 가장 첫번째에 멀쩡한 원본 출력
    rows_augmenters = [
        ("Fliplr", [(str(p), iaa.Fliplr(p)) for p in [0, 1]]),
        # ("Add", [("value=%d" % (val,), iaa.Add(val)) for val in [-45, -25, 25, 45]]),
        # ("Multiply", [("value=%.2f" % (val,), iaa.Multiply(val)) for val in [0.5, 0.8, 1.2, 1.5]]),
        # ("GaussianBlur", [("sigma=%.2f" % (sigma,), iaa.GaussianBlur(sigma=sigma)) for sigma in [0.25, 0.50]]),
        # ("Emboss\n(alpha=1)", [("strength=%.2f" % (0,), iaa.Emboss(alpha=1, strength=0))]),
        # ("ContrastNormalization", [("alpha=%.1f" % (alpha,), iaa.ContrastNormalization(alpha=alpha)) for alpha in [0.75, 1.25]]),
        # ("Grayscale", [("alpha=%.1f" % (1.0,), iaa.Grayscale(alpha=1.0))])
    ]

        # print("[draw_per_augmenter_images] Augmenting...")

    output_cnt = 0
    for i in range(nSample):
        cnt = 0
        for (row_name, augmenters) in rows_augmenters:
            for img_title, augmenter in augmenters:
                aug_det = augmenter.to_deterministic()
                # row_images.append(aug_det.augment_image(image))
                aug_img = aug_det.augment_image(m_training_img[i])
                # buf = "id_%04d_%02d.jpg" % (i, cnt)
                # plt.imsave(buf, aug_img)
                output_img[output_cnt] = np.float32(aug_img)
                cnt += 1
                output_cnt += 1

    return output_img
