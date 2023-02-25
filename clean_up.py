from method import records, data
import cv2
import os.path
from tqdm import tqdm

input_dir = os.path.join(data.dataset_dir, records['clean-up']['input'])
output_dir = os.path.join(data.dataset_dir, records['clean-up']['output'])

img_paths = os.listdir(input_dir)
for filename in tqdm(os.listdir(input_dir), desc='generating'):
    if filename.split('.')[-1] not in data.image_formats:
        continue
    img = data.read_img(os.path.join(input_dir, filename))[0, :, :, ::-1].numpy()
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)[:, :, ::-1]
    data.write_img(os.path.join(output_dir, filename), img)
