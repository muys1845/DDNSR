from method import data, gan, networks, layers, dir_name
import os.path
import tensorflow as tf
import argparse


class Model(gan.Loss, gan.GAN):
    def __init__(self, source_name: str, target_type: str, up_type='esrgan', down_type='dual', content=(3, 4),
                 cycle=(2, 2), weights=(0.1, 1.0, 10.0), schedule=(50, 100, 200, 300), train=True):

        assert source_name in ['track1', 'track2']
        assert up_type in ['esrgan', 'psnr']
        assert down_type in ['dual', 'single']

        self.up = networks.reconstruction_network()
        if source_name == 'track1':
            disc_up, self.disc_up = 'vggstyle', networks.discriminator_vggstyle(64, 5)
        else:
            disc_up, self.disc_up = 'patchgan', networks.discriminator_patchgan()
        self.down = networks.degradation_network(input_type=down_type)
        self.disc_down = networks.discriminator_vggstyle(128, 3)

        weights = [float(x) for x in weights]
        self.w1, self.w2, self.w3 = weights

        if up_type == 'esrgan':
            self.train_step = self.esrgan
            models = [self.up, self.down, self.disc_up, self.disc_down]
            history_items = ['content_up', 'adv_up', 'cycle_up', 'adv_disc_up',
                             'content_down', 'adv_down', 'cycle_down', 'adv_disc_down']
        else:
            self.train_step = self.psnr
            models = [self.up, self.down, self.disc_down]
            history_items = ['loss_up', 'content_down', 'adv_down', 'cycle_down', 'adv_disc_down']

        self.feature_content = layers.vgg_feature_model(*content)
        self.feature_cycle = layers.vgg_feature_model(*cycle)

        model_dir = '{}_{}_{}_content{}{}_cycle{}{}_{}_{}_{}_{}_input_{}k'.format(target_type, up_type, disc_up,
                                                                                  *content, *cycle, *weights,
                                                                                  down_type, schedule[-1])
        saved_dir = os.path.join(dir_name, source_name, model_dir)
        super(Model, self).__init__(saved_dir, models, history_items)

        if train:
            self.train(data.train_data(source_name, target_type), schedule=schedule)
            self.up.save(os.path.join(self.saved_dir, 'reconstruction-network.h5'))
            self.down.save(os.path.join(self.saved_dir, 'degradation-network.h5'))

    def content_loss(self, real, fake):
        return self.mse(self.feature_content(layers.downscale(real, 4)), self.feature_content(fake))

    def cycle_loss(self, real, fake):
        return self.mse(self.feature_cycle(real), self.feature_cycle(fake))

    def train_step(self, data_batch):
        raise NotImplementedError

    def save_temp_results(self, data_batch):
        lores_x_real, hires_y_real = data_batch

        hires_x_fake = self.up(lores_x_real)
        hires_x_static = layers.upsample(lores_x_real)
        lores_x_fake = self.down([hires_x_fake, lores_x_real])

        lores_y_fake = self.down([hires_y_real, lores_x_real])
        lores_y_static = layers.downscale(hires_y_real)
        hires_y_fake = self.up(lores_y_fake)

        titles = ['real', 'fake_fake', 'static', 'fake']
        for item in zip(['x', 'y'], [lores_x_real, hires_y_real], [lores_x_fake, hires_y_fake],
                        [hires_x_static, lores_y_static], [hires_x_fake, lores_y_fake]):
            for i, imgs in enumerate(zip(*item[1:]), start=1):
                self.savefig('{}_{}'.format(i, item[0]), imgs, titles, (2, 2))

    def psnr(self, data_batch):
        with tf.GradientTape(persistent=True) as tape:
            lores_x_real, hires_y_real = data_batch

            hires_x_fake = self.up(lores_x_real)
            lores_x_fake = self.down([hires_x_fake, lores_x_real])
            lores_y_fake = self.down([hires_y_real, lores_x_real])
            hires_y_fake = self.up(lores_y_fake)

            content_down = self.content_loss(hires_y_real, lores_y_fake)
            cycle_down = self.cycle_loss(lores_x_real, lores_x_fake)

            score_lores_real = self.disc_down(lores_x_real)
            score_lores_fake = self.disc_down(lores_y_fake)

            loss_up = self.mse(hires_y_real, hires_y_fake)
            adv_down, adv_disc_down = self.dcgan_loss(score_lores_real, score_lores_fake)
            loss_down = content_down * self.w1 + adv_down * self.w2 + cycle_down * self.w3
        for loss, model in zip([loss_up, loss_down, adv_disc_down], [self.up, self.down, self.disc_down]):
            grad = tape.gradient(loss, model.trainable_variables)
            self.optm.apply_gradients(zip(grad, model.trainable_variables))
        return loss_up, content_down, adv_down, cycle_down, adv_disc_down

    def esrgan(self, data_batch):
        with tf.GradientTape(persistent=True) as tape:
            lores_x_real, hires_y_real = data_batch

            hires_x_fake = self.up(lores_x_real)
            lores_x_fake = self.down([hires_x_fake, lores_x_real])
            lores_y_fake = self.down([hires_y_real, lores_x_real])
            hires_y_fake = self.up(lores_y_fake)

            content_up = self.vgg54_loss(hires_y_real, hires_y_fake)
            cycle_up = self.mae(hires_y_real, hires_y_fake)
            content_down = self.content_loss(hires_y_real, lores_y_fake)
            cycle_down = self.cycle_loss(lores_x_real, lores_x_fake)

            score_hires_real = self.disc_up(hires_y_real)
            score_hires_fake = self.disc_up(hires_y_fake)
            score_lores_real = self.disc_down(lores_x_real)
            score_lores_fake = self.disc_down(lores_y_fake)

            adv_up, adv_disc_up = self.relativistic_loss(score_hires_real, score_hires_fake)
            loss_up = content_up + adv_up * 0.005 + cycle_up * 0.01
            adv_down, adv_disc_down = self.dcgan_loss(score_lores_real, score_lores_fake)
            loss_down = content_down * self.w1 + adv_down * self.w2 + cycle_down * self.w3

        for loss, model in zip([loss_up, loss_down, adv_disc_up, adv_disc_down],
                               [self.up, self.down, self.disc_up, self.disc_down]):
            grad = tape.gradient(loss, model.trainable_variables)
            self.optm.apply_gradients(zip(grad, model.trainable_variables))
        return content_up, adv_up, cycle_up, adv_disc_up, content_down, adv_down, cycle_down, adv_disc_down


def main():
    # main
    Model('track1', 'div2k')
    Model('track2', 'cleanup')


def ablation_study():
    # three parts of weights
    for i in range(3):
        weights = [0.1, 1.0, 10.0]
        weights[i] = 0.
        Model('track1', 'div2k', weights=weights, schedule=(50, 100))

    # dual input or not
    for down_type in ['dual', 'single']:
        Model('track1', 'div2k', down_type=down_type, schedule=(50, 100))


def different_targets():
    for target_type in ['cleanup', 'canon', 'mixed', 'div2k']:
        Model('track2', target_type)


def quantitative_model():
    # appendix
    Model('track1', 'div2k', up_type='psnr')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-source_name', default='', type=str)
    parser.add_argument('-target_type', default='', type=str)
    args = parser.parse_args()
    Model(args.source_name, args.target_type)
