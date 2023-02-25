import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import json

tf.compat.v1.logging.set_verbosity('ERROR')

dir_name = os.path.dirname(os.path.dirname(__file__))
""" Absolute path of the project directory. """

record_path = os.path.join(dir_name, 'records.json')
assert os.path.exists(record_path), 'records file doesn\'t exist'
with open(record_path, 'r') as f:
    records = json.load(f)
