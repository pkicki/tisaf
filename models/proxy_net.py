from math import pi
import tensorflow as tf
from models.submodels import MapFeaturesProcessor, FeatureExtractorLayer, EstimatorLayer

tf.enable_eager_execution()


class ProxyNet(tf.keras.Model):

    def __init__(self):
        super(ProxyNet, self).__init__()

    def update_weights(self, model, tau):
        for i in range(len(self.trainable_variables)):
            tf.assign(self.trainable_variables[i],
                      tau * model.trainable_variables[i] + (1. - tau) * self.trainable_variables[i])

