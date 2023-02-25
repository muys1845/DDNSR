import os
from method import dir_name, records
import tensorflow as tf
import glob
import random

dataset_dir = os.path.join(dir_name, 'datasets')
image_formats = ['jpg', 'jpeg', 'png', 'bmp']


@tf.function
def read_img(path):
    img = tf.io.decode_image(tf.io.read_file(path))[tf.newaxis]
    return tf.cast(img, tf.float32) / 255.


def write_img(img_path, img):
    if len(img.shape) == 4:
        img = img[0]
    img = tf.clip_by_value(img, 0., 1.)
    img = tf.cast(img * 255., tf.uint8)
    img = tf.io.encode_png(img, compression=0)
    tf.io.write_file(img_path, img)


def list_img(dir_):
    img_paths = []
    for image_format in image_formats:
        img_paths += glob.glob(os.path.join(dataset_dir, dir_, '*.' + image_format))
    return img_paths


def dataset(data_name: str, crop: tuple):
    @tf.function
    def crop_image(image):
        image = tf.image.random_flip_left_right(image)[tf.newaxis]
        imgs = []
        for _ in range(16):
            imgs.append(tf.image.random_crop(image, (1, *crop, 3)))
        imgs = tf.concat(imgs, axis=0)
        return imgs

    @tf.function
    def to_float(image):
        return tf.cast(image, tf.float32) / 255.

    assert data_name in records['train'], 'dataset {} not found in: records.json'.format(data_name)
    dirs = records['train'][data_name]
    if type(dirs) is str:
        dirs = [dirs]

    all_image_paths = []
    for dir_ in dirs:
        all_image_paths += list_img(dir_)

    assert all_image_paths, 'no image found, recheck your file: records.json'
    random.shuffle(all_image_paths)
    return tf.data.Dataset.from_tensor_slices(all_image_paths).map(tf.io.read_file).map(tf.io.decode_image).map(
        crop_image).map(to_float).unbatch().repeat()


def train_data(source_name, target_type):
    assert source_name in ['track1', 'track2']
    source = dataset(source_name + '-source', crop=(32, 32))
    target = dataset('{}-target-{}'.format(source_name, target_type), crop=(128, 128))
    return tf.data.Dataset.zip((source, target))
