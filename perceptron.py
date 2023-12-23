"""
components needed
- a function to specify the number of neurons
- whether or not to set a threshold
- a matrix of weights
- training set

"""

import random
import sys
import time

# possible X values
X_OPTIONS = [-1, 1]  # TODO: should it be -1, 1?

# should be able to set weights to random initially
INITIAL_WEIGHT = 0.5

INITIAL_THRESHOLD = 2
DEFAULT_RATE = 0.05
DEFAULT_NEURONS = 5
DEFAULT_SAMPLE_SIZE = 100
DEFAULT_TEST_SIZE = 20


def dot(x, y):
  # validation
  if (len(x) != len(y)):
    raise ValueError('L17: Dot product cannot be computed on\
                           vectors of different lengths')
  return sum([x[i] * y[i] for i in range(len(x))])


def accuracy(y_expected, y_actual):
  if (len(y_expected) != len(y_actual)):
    raise ValueError('L24: Cannot computed accuracy when \
                      expected and actual arrays are different length')

  return (len([
      y_expected[i]
      for i in range(len(y_expected)) if y_expected[i] == y_actual[i]
  ]) / len(y_expected))


def generate_vector(length, options=X_OPTIONS):
  return [random.choice(options) for i in range(length)]


# returns a tuple X_values, Y_values
def generate_dataset(func, x_len=DEFAULT_NEURONS, length=DEFAULT_SAMPLE_SIZE):
  X_values = []
  Y_values = []

  for _ in range(length):
    X = generate_vector(x_len)
    X_values.append(X)
    Y_values.append(func(X))

  return X_values, Y_values


class Perceptron:

  def __init__(self, num_neurons=DEFAULT_NEURONS):
    # params
    # - initial weights
    # - initial threshold
    # - learning rate
    self.weights = [INITIAL_WEIGHT] * num_neurons
    self.threshold = INITIAL_THRESHOLD
    self.learning_rate = DEFAULT_RATE

  def compute_sample(self, input):
    # TODO: consider the threshold
    # TODO: should we consider a bias term?
    return 1 if dot(input, self.weights) > 0 else -1

  def learn_sample(self, input, expected):
    prod = dot(input, self.weights)

    result = 1 if prod > self.threshold else -1

    # update weight if the result diverges from expected
    # TODO: should we update weights regardless? (I don't think so)
    if result != expected:
      self.weights = [
          self.weights[i] + self.learning_rate * input[i] * expected
          for i in range(len(self.weights))
      ]

  def train(self, train_X, train_Y, log_steps=False, num_passes=1):
    # validation
    if (len(train_X) != len(train_Y)):
      raise ValueError('L50: number of training samples is not \
                        equal to number of labels')

    for n in range(num_passes):
      if log_steps:
        print(f'For pass #{n+1}')
      for i in range(len(train_X)):
        self.learn_sample(train_X[i], train_Y[i])
        if log_steps:
          # out = f'n={i}\t'
          formatted_weights = '\t\t'.join(f'{w:.2f}' for w in self.weights)
          print(f'\rn={i+1:03}\t\t[{formatted_weights}]', end='')
          sys.stdout.flush()
          time.sleep(0.05)

      if log_steps:
        # newline
        print()

  def accuracy(self, test_X, test_Y):
    if (len(test_X) != len(test_Y)):
      raise ValueError('L61: number of test samples is not \
                        equal to number of labels')

    result_Y = []
    for i in range(len(test_X)):
      result_Y.append(self.compute_sample(test_X[i]))

    return accuracy(result_Y, test_Y)


# just map x[0]
def function1(x):
  return 1 if x[0] > 0 else -1


# take some of x[0] and x[1]
def function2(x):
  return 1 if x[0] + x[1] > 10 else -1


# take total sum
def function3(x):
  return 1 if sum(x) > 20 else -1


# just map x[1]
def function4(x):
  return 1 if x[1] > 0 else -1


def function5(x):
  return 1 if x[0] > 0 and x[1] > 0 else -1


def test1():
  X, Y = generate_dataset(function4)

  (X_train, X_test, Y_train,
   Y_test) = (X[:DEFAULT_SAMPLE_SIZE - DEFAULT_TEST_SIZE],
              X[DEFAULT_SAMPLE_SIZE - DEFAULT_TEST_SIZE:],
              Y[:DEFAULT_SAMPLE_SIZE - DEFAULT_TEST_SIZE],
              Y[DEFAULT_SAMPLE_SIZE - DEFAULT_TEST_SIZE:])

  p = Perceptron()

  print(f'Train acuracy before training {p.accuracy(X_train, Y_train):.2f}')
  print(f'Test accuracy before training {p.accuracy(X_test, Y_test):.2f}')

  p.train(X_train, Y_train, True, 2)

  print(f'Train acuracy after training {p.accuracy(X_train, Y_train):.2f}')
  print(f'Test accuracy after training {p.accuracy(X_test, Y_test):.2f}')


test1()
