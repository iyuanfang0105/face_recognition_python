import os
import random
import numpy as np
from scipy import misc
from PIL import Image
from matplotlib import pyplot as plt


def imread(path):
    """
    read image
    :param path: image path
    :return: RGB image ndarray (h, w, 3)
    """
    img = misc.imread(path).astype(np.float32)
    if len(img.shape) == 2:
        img = np.dstack((img, img, img))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def imread_from_folder(path, partion=1, shuffle=False):
    """
    read images from folder
    :param path: folder path
    :param partion: read partion of total (0, 1)
    :return: a list contains RGB images ndarray (h, w, 3)
    """
    images_file_name = os.listdir(path)
    images_path = [os.path.join(path, file_name) for file_name in images_file_name]
    read_iamges_num = int(round(partion * len(images_path)))

    if shuffle:
        images_path = random.sample(images_path, read_iamges_num)
    else:
        images_path = images_path[:read_iamges_num]

    images = []
    for image_path in images_path:
        images.append(imread(image_path))
    return images


def imsave(path, img):
    """
    save image
    :param path: image path saved
    :param img: image data (ndarray)
    :return: null
    """
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=95)


def imshow(img):
    """
    show image
    :param img: image data (ndarray)
    :return: null
    """
    misc.imshow(img)


def show_images(images, cols=1, titles=None, axis='off'):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    # if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    if titles is None: titles = ['%d' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        if image.max() > 1.0:
            temp = image / 255.0
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        a.axis(axis)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(temp)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


if __name__ == '__main__':
    # test_image_path = '/home/meizu/WORK/code/neural_style/images/1-content.jpg'
    # image = imread(test_image_path)
    # misc.imshow(image)
    # print ''

    image_folder_path = '../data/COCO/val2014'
    images = imread_from_folder(image_folder_path, partion=0.01, shuffle=True)
    show_images(images[::20], cols=3)
    print len(images)
