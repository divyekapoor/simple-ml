# Learn a dataset that is modeled by a polynomial using SGD.
#
# We start with a model that is able to represent all possible polynomials up
#  to degree 10.
#
# Model: y = w_i x_i^i + w_0 for i in [0, 10]
# w_i being model weights.
#
# This model is capable of representing all possible lines and thus,
# it should be able to learn our simple linear model: y = 2 x + 7
# However, it can suffer from interference from higher order terms.
#
# Tested with Python 3.5
# $ python3 --version
# Python 3.5.2+
#

import math
import random

import tensorflow as tf
from tensorflow.contrib import layers, learn
from tensorflow.python.training.ftrl import FtrlOptimizer
from typing import List, Iterable

from datagen import training_datagen, Entry, test_datagen, \
    random_entry_from_training_data, categorical_feature


class Model(object):
    def __init__(self, degree: int):
        self.degree = degree
        self.training_datagen = [entry for entry in training_datagen()]
        self.test_datagen = [entry for entry in test_datagen()]
        self.training_values = None  # type: tf.Constant
        self.training_labels = None  # type: tf.Constant
        self.test_values = None  # type: tf.Constant
        self.test_labels = None  # type: tf.Constant
        self.weights = None  # type: tf.Constant
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.weights = tf.constant(
                [random.random() for _ in range(len(self.training_datagen))]
            )
            with tf.variable_scope("training_data") as training_scope:
                self.training_values = tf.constant(
                    [entry.value for entry in self.training_datagen],
                    name="training_values",
                    dtype=tf.float32,
                    shape=[len(self.training_datagen)],
                    verify_shape=True)
                self.training_labels = tf.constant(
                    [entry.label for entry in self.training_datagen],
                    name="training_labels",
                    dtype=tf.float32,
                    shape=[len(self.training_datagen)],
                    verify_shape=True)
            with tf.variable_scope("test_data") as test_scope:
                self.test_values = tf.constant(
                    [entry.value for entry in self.test_datagen],
                    name="test_values",
                    dtype=tf.float32,
                    shape=[len(self.test_datagen)],
                    verify_shape=True)
                self.test_labels = tf.constant(
                    [entry.label for entry in self.test_datagen],
                    name="test_label",
                    dtype=tf.float32,
                    shape=[len(self.test_datagen)],
                    verify_shape=True)
            self.init_op = tf.global_variables_initializer()

    def _training_data(self):
        return {
                   "value": self.training_values,
                "weights": self.weights
               }, self.training_labels

    def _test_data(self):
        return {
                   "value": self.test_values,
               }, self.test_labels

    def predict(self, x: Entry) -> float:
        """Evaluate the model and make a prediction. We reserve the last 
        weight as a penalty on model weights for regularization. However, 
        it is not used in prediction."""
        return self.linear_regressor.predict(x=x.value)

    def train(self, max_steps: int, learning_rate: float,
              learning_rate_power: float, l1_regularization_penalty: float,
              l2_regularization_penalty: float):
        """Train the model using the Follow-the-regularized-leader optimizer 
        which improves upon the basic stochastic gradient descent gradient 
        update function by taking into account the fact that while getting 
        the gradients, we should be lazy in computing the regularization 
        penalty to after the weight update."""
        with self.graph.as_default():
            self.linear_regressor = learn.LinearRegressor(
                feature_columns=[layers.real_valued_column("value")],
                optimizer="SGD")
            self.linear_regressor.fit(
                input_fn=self._training_data,
                max_steps=max_steps,
                monitors=[])

    def test(self) -> dict:
        return self.linear_regressor.evaluate(
            input_fn=self._test_data,
            steps=1)

    def __str__(self):
        return 'Model()'


if __name__ == '__main__':
    degree = 3
    iterations = 5000
    learning_rate = 0.01
    learning_rate_power = -0.6
    l1_regularization_penalty = 0.
    l2_regularization_penalty = 0.001

    model = Model(degree)
    model.train(
        iterations, learning_rate, learning_rate_power,
        l1_regularization_penalty, l2_regularization_penalty)

    loss = model.test()
    print("Final model loss: {}", loss)
