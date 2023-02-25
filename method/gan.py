import os
import json
import time
from method import layers
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


class Loss:
    vgg54 = layers.vgg_feature_model(5, 4)
    mse = keras.losses.MeanSquaredError()
    mae = keras.losses.MeanAbsoluteError()

    @staticmethod
    def bce(value, y_pred):
        """ Binary crossentropy. """
        return tf.reduce_mean(keras.losses.binary_crossentropy(tf.fill(y_pred.shape, value), y_pred, from_logits=True))

    @classmethod
    def vgg54_loss(cls, real, fake):
        return cls.mse(cls.vgg54(real), cls.vgg54(fake))

    @classmethod
    def relativistic_loss(cls, score_real, score_fake):
        disc_real = score_real - tf.reduce_mean(score_fake)
        disc_fake = score_fake - tf.reduce_mean(score_real)
        loss_gen = cls.bce(1., disc_fake) + cls.bce(0., disc_real)
        loss_disc = cls.bce(1., disc_real) + cls.bce(0., disc_fake)
        return loss_gen, loss_disc

    @classmethod
    def dcgan_loss(cls, score_real, score_fake):
        loss_gen = cls.bce(1., score_fake)
        loss_disc = cls.bce(1., score_real) + cls.bce(0., score_fake)
        return loss_gen, loss_disc


class GAN:
    def __init__(self, saved_dir, models, history_items):
        print('saved_dir: ' + saved_dir)
        self.saved_dir = saved_dir
        self.temp_history_dir = os.path.join(saved_dir, 'temp_history')
        self.temp_results_dir = os.path.join(saved_dir, 'temp_results')
        self.history_files_dir = os.path.join(saved_dir, 'history_files')
        for dir_ in [self.saved_dir, self.temp_history_dir, self.temp_results_dir, self.history_files_dir]:
            os.makedirs(dir_, exist_ok=True)

        history_record_path = os.path.join(self.history_files_dir, 'items.json')
        if not os.path.exists(history_record_path):
            with open(history_record_path, 'w') as f:
                json.dump(history_items, f, indent=2)
        self.logging_path = os.path.join(self.saved_dir, 'training_logging.json')
        if not os.path.exists(self.logging_path):
            with open(self.logging_path, 'w') as f:
                json.dump({'batch_cur': 0, 'wall_time': 0}, f, indent=2)

        self.models = models
        ckpt_dict = {'model_{}'.format(ind): model for ind, model in enumerate(models, start=1)}
        self.ckpt = tf.train.Checkpoint(**ckpt_dict)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, os.path.join(saved_dir, 'checkpoints'), max_to_keep=2)
        self.history_items = history_items
        self.optm = None

    def savefig(self, filename, imgs, titles, subplot):
        assert len(imgs) == len(titles) <= subplot[0] * subplot[1]
        for i, item in enumerate(zip(imgs, titles), start=1):
            img = tf.clip_by_value(item[0], 0., 1.)
            plt.subplot(*subplot, i)
            plt.imshow(img)
            plt.axis('off')
            plt.title('{} - {}'.format(item[1], img.shape[:-1]))
        plt.savefig(os.path.join(self.temp_results_dir, '{}.jpg'.format(filename)), dpi=200)
        plt.close()

    def save_temp_results(self, data_batch):
        raise NotImplementedError

    def train_step(self, data_batch):
        raise NotImplementedError

    def train(self, train_data, lr=1e-4, batch_size=16, schedule=(50, 100, 200, 300), save_per_kbatch=5):
        with open(self.logging_path, 'r') as f:
            record = json.load(f)
        batch_cur, wall_time = record['batch_cur'], record['wall_time']
        if batch_cur and self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print('latest checkpoint restored')
        else:
            batch_cur = wall_time = 0
            print('training in the first time')

        train_data = train_data.shuffle(1024, reshuffle_each_iteration=True).batch(batch_size).as_numpy_iterator()
        # shuffle to aviod all batch images are cropped from one single image
        period_batch = save_per_kbatch << 10

        start = time.time()
        for phase, total_batch in enumerate([x << 10 for x in schedule], start=1):
            print('training on phase {}/{} - learning rate: {}'.format(phase, len(schedule), lr))
            self.optm = keras.optimizers.Adam(lr)
            func = tf.function(self.train_step)

            while batch_cur < total_batch:
                history = np.zeros([period_batch, len(self.history_items)], dtype=np.float32)
                for batch in range(period_batch):
                    history[batch, :] = func(next(train_data))
                    batch_cur += 1
                    print('\rbatch: {}/{}'.format(batch_cur, total_batch), end='')
                history = np.transpose(history, (1, 0))

                for loss, label in zip(history, self.history_items):
                    plt.plot(loss, label=label)
                    plt.legend()
                    plt.savefig(os.path.join(self.temp_history_dir, '{}.jpg'.format(label)))
                    plt.close()

                np.save(os.path.join(self.history_files_dir, '{}k.npy'.format(batch_cur >> 10)), history)
                self.save_temp_results(next(train_data))
                self.ckpt_manager.save()
                with open(self.logging_path, 'w') as f:
                    json.dump({'batch_cur': batch_cur, 'wall_time': int(wall_time + time.time() - start)}, f, indent=2)

            print(' - complete')
            lr /= 2

        sec = time.time() - start + wall_time
        print('wall time: %dh %dm %ds\n' % (sec / 3600, sec % 3600 / 60, sec % 60))
