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
import random
from itertools import accumulate

from typing import List, Iterable

from datagen import training_datagen, Entry, test_datagen, \
    random_entry_from_training_data


class Model(object):
    def __init__(self, degree):
        self.weights = [random.random() for _ in range(0, degree + 1)]
        self.training_loss = float(0)
        self.test_loss = float(0)

    def feature(self, degree: int, entry: Entry) -> float:
        return entry.value ** degree

    def predict(self, x: Entry) -> float:
        """Evaluate the model and make a prediction. We reserve the last 
        weight as a penalty on model weights for regularization. However, 
        it is not used in prediction."""
        value = float(0)
        for i in range(len(self.weights)):
            value += self.weights[i] * self.feature(i, x)
        return value

    def evolve(self,
               entry: Entry,
               learning_rate: List[float],
               regularization_rate: float,
               training_data: Iterable[Entry],
               test_data: Iterable[Entry]):
        """Evolve the model parameters based on a training dataset example.
        
        This is the most important method for Stochastic Gradient Descent: 
        the computation of gradients to update the weights is crucial for 
        convergence. Once again, accuracy of implementation here is crucial 
        for convergence.
        
        In this method, we are using the simple stochastic gradient update 
        algorithm:
        
        W(t+1) = W(t) - learning_rate * derivative(Loss(W(t)), w)
        
        Since our model has a loss function:
        Loss(W(t)) = (w_0 + w_1 x + w_2 x^2 + w_3 x^3... - actual_value)^2 
        + λ(w_0^2 + w_1^2 + w_2^2 ...)
                
        And thus we come to our update equations that are implemented in this 
        method.
        
        :argument entry A random example from the training set.
        :argument learning_rate A pair of floats that control how slowly we 
        will move the slope and bias parameters to reduce errors.
        :argument regularization_rate Penalty for high weights
        :argument training_data 
        :argument test_data
        """
        predict_error = (self.predict(entry) - entry.label)

        # Update weights
        for i in range(len(self.weights)):
            learning_rate_coefficient = learning_rate[i]
            self.weights[i] -= learning_rate_coefficient * (
                    2 * self.feature(i, entry) * predict_error
                    + 2 * regularization_rate * self.weights[i])

        # The evaluation of these two values is not required at each update
        # step, but it's helpful to watch. Usually, these evaluations are
        # done once every N training steps. (These methods are expensive to
        # evaluate).
        self.training_loss = self.get_loss(training_data)
        self.test_loss = self.get_loss(test_data)

    def get_loss(self, eval_data: Iterable[Entry]) -> float:
        """A euclidean loss function: (predict(x) - y)^2. This runs over an 
        entire dataset and tells us of the quality of this model when run 
        over that dataset. Useful for judging how a model is performing 
        during training runs.
        
        This method is expensive to run, so we may want to run is less 
        frequently during training.
        """
        loss = float(0)
        count = 0
        for entry in eval_data:
            loss += (self.predict(entry) - entry.label) ** 2
            count += 1
        return loss / count

    def regularization_feature(self) -> float:
        result = float(0)
        for weight in self.weights:
            result += weight ** 2
        return result

    def __str__(self):
        return 'Model(weights: {}, training_loss: {}, test_loss: {})' \
            .format(self.weights, self.training_loss, self.test_loss)


def train(degree: int,
          iterations: int,
          learning_rate: List[float],
          regularization_rate: float) -> Model:
    model = Model(degree)
    for i in range(iterations):
        print("Model: {}".format(model))
        model.evolve(random_entry_from_training_data(),
                     learning_rate,
                     regularization_rate,
                     training_datagen(),
                     test_datagen())
        if model.test_loss < 1e-5:
            print("Number of iterations: {}", i)
            return model
    return model


if __name__ == '__main__':
    # Take a second degree polynomial model and learn its coefficients.
    # Example result: Model(weights: [6.988261960152755, 2.011126393503278,
    # -0.002081713831478238], training_loss: 1.773968066101656e-05,
    # test_loss: 1.794096926436063e-05)
    #
    # for the given underlying model: 7 + 2x.
    #
    # This is a pretty good learnt function.
    #
    # However, the largest tweaking required was with learning rates here to
    # get the model to converge.
    #
    degree = 2
    iterations = 5000
    learning_rate = [0.01, 0.01, 0.0005]
    regularization_rate = 0.00001
    trained_model = train(degree, iterations, learning_rate,
                          regularization_rate)
    print("Final model is: {}", trained_model)
