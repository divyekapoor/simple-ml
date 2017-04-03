# simple-ml
Stochastic Gradient Descent to learn simple models.

# Basic Underlying Model

The linear regression training algorithms in this repository are trying
to learn the function: 2x + 7.

Ideally, this function should be learned with very little error since
this is a linear function with 2 weights. The challenge is to do it
quickly. To this end, several algorithms are used to hasten the learning
process:

 1. A simple SGD with a fixed learning rate.
 1. A more general SGD that represents the model as a general polynomial.
 1. A SGD with an annealing learning rate schedule.
 1. A SGD with an inverse Hessian learning rate.
 1. A SGD with AdaGrad controlling the learning rate.
 1. A SGD with AdaDelta controlling the learning rate.
 1. A SGD with RMSProp controlling the learning rate.

# Installation instructions

   ```sh
   $ virtualenv -p python3 venv
   $ source venv/bin/activate
   $ pip3 install -U -r requirements.txt
   ```

# License
MIT
