from method import layers, data
from tensorflow import keras
import os
from tqdm import tqdm
import argparse


def generate_full_results(model_path, input_dir, output_dir):
    model = keras.models.load_model(
        model_path, custom_objects=layers.custom_objects)

    input_dir = os.path.join(input_dir)
    output_dir = os.path.join(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for filename in tqdm(sorted(os.listdir(input_dir)), desc='generating'):
        img = data.read_img(os.path.join(input_dir, filename))
        data.write_img(os.path.join(output_dir, filename), model(img))


class Model:
    """An api for non-tensorflow users"""

    def __init__(self, model_path):
        self.model = keras.models.load_model(
            model_path, custom_objects=layers.custom_objects)

    def __call__(self, img, *args, **kwargs):
        """
        input: ndarray of shape (b, h, w, c)
        output: ndarray of shape (b, h*4, w*4, c)
        """
        return self.model(img).numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_path', default='', type=str)
    parser.add_argument('-input_dir', default='', type=str)
    parser.add_argument('-output_dir', default='', type=str)
    args = parser.parse_args()
    generate_full_results(args.model_path, args.input_dir, args.output_dir)
