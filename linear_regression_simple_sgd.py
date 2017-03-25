# Learn a dataset that is modeled by a simple linear fit using SGD.
#
# We start with a model that is able to represent all possible lines.
#
# Model: y = mx + b
# m being a free model parameter that represents slope of a line
# b being a free model parameter that represents the intercept of a line
#
# This model is capable of representing all possible lines and thus,
# it should be able to learn our simple linear model: y = 2 x + 7
#
# Tested with Python 3.5
# $ python3 --version
# Python 3.5.2+
#
import random

from typing import List, Iterable

from datagen import training_datagen, Entry, test_datagen, \
    random_entry_from_training_data


class Model(object):
    def __init__(self, slope: float, bias: float):
        self.slope = slope
        self.bias = bias
        self.training_loss = float(0)
        self.test_loss = float(0)

    def predict(self, x: Entry):
        """Evaluate the model and make a prediction."""
        return self.slope * x.value + self.bias

    def evolve(self, entry: Entry, learning_rate: List[float],
               training_data: Iterable[Entry], test_data: Iterable[Entry]):
        """Evolve the model parameters based on a training dataset example.
        
        This is the most important method for Stochastic Gradient Descent: 
        the computation of gradients to update the weights is crucial for 
        convergence. Once again, accuracy of implementation here is crucial 
        for convergence.
        
        In this method, we are using the simple stochastic gradient update 
        algorithm:
        
        W(t+1) = W(t) - learning_rate * derivative(Loss(W(t)), w)
        
        Since our model has a loss function:
        Loss(W(t)) = (slope * x + bias - actual_value)^2
        
        The derivative of the loss function with respect to the weights is:
        derivative(Loss w.r.t slope) = 2 * x * (slope * x + bias - actual_value)
        derivative(Loss w.r.t bias) = 2 * (slope * x + bias - actual_value)
        
        And thus we come to our update equations that are implemented in this 
        method.
        
        :argument entry A random example from the training set.
        :argument learning_rate A pair of floats that control how slowly we 
        will move the slope and bias parameters to reduce errors.
        :argument training_data 
        :argument test_data
        """
        predict_error = (self.predict(entry) - entry.label)
        self.slope -= learning_rate[0] * 2 * entry.value * predict_error
        self.bias -= learning_rate[1] * 2 * predict_error

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

    def __str__(self):
        return 'Model(slope: {}, bias: {}, training_loss: {}, test_loss: {})' \
            .format(self.slope, self.bias, self.training_loss, self.test_loss)


def train(iterations: int, learning_rate: List[float]) -> Model:
    model = Model(random.random(), random.random())

    for i in range(iterations):
        print("Model: {}".format(model))
        model.evolve(random_entry_from_training_data(),
                     learning_rate,
                     training_datagen(),
                     test_datagen())
        if model.test_loss < 1e-5:
            print("Number of iterations: {}", i)
            return model
    return model


if __name__ == '__main__':
    trained_model = train(10000, [0.05, 0.05])
    print("Final model is: {}", trained_model)
