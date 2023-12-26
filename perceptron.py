"""
Current status:
- the perceptron learns well when the inputs are x = -1 or 1
  - but this is limited because we can't learn interesting functions
- however if the x values are a range between (-30, 30) then
  the weights are unstable
- so need to add a concept of "momentum" to the weights

"""

import random
import sys
import time


# possible X values
def DEFAULT_GENERATOR():
  return random.choice([-1, 1])


# should be able to set weights to random initially
INITIAL_WEIGHT = 0.5

INITIAL_THRESHOLD = 0
DEFAULT_RATE = 0.01
DEFAULT_NEURONS = 5
DEFAULT_SAMPLE_SIZE = 1000
DEFAULT_TEST_SIZE = 20  # a fraction of the sample size
DEFAULT_PASSES = 2


def dot(x, y):
  # validation
  if (len(x) != len(y)):
    raise ValueError('L17: Dot product cannot be computed on\
                           vectors of different lengths')
  return sum([x[i] * y[i] for i in range(len(x))])

def unit(x):
  return 0 if x == 0 else 1 if x > 0 else -1


def accuracy(y_expected, y_actual):
  if (len(y_expected) != len(y_actual)):
    raise ValueError('L24: Cannot computed accuracy when \
                      expected and actual arrays are different length')

  return (len([
      y_expected[i]
      for i in range(len(y_expected)) if y_expected[i] == y_actual[i]
  ]) / len(y_expected))


# returns a tuple X_values, Y_values
def generate_dataset(func,
                     x_len=DEFAULT_NEURONS,
                     length=DEFAULT_SAMPLE_SIZE,
                     generator=DEFAULT_GENERATOR):
  X_values = []
  Y_values = []

  for _ in range(length):
    X = [generator() for i in range(x_len)] + [1]  # bias term
    X_values.append(X)
    Y_values.append(func(X))

  return X_values, Y_values


class Perceptron:

  def __init__(self, num_neurons=DEFAULT_NEURONS):
    # params
    # - initial weights
    # - initial threshold
    # - learning rate
    self.weights = ([INITIAL_WEIGHT] * num_neurons + 
                    [INITIAL_WEIGHT])  # bias term
    self.threshold = INITIAL_THRESHOLD
    self.learning_rate = DEFAULT_RATE

  def compute_sample(self, input):
    return 1 if dot(input, self.weights) > self.threshold else -1

  def learn_sample(self, input, expected):
    result = self.compute_sample(input)

    # update weight if the result diverges from expected
    # TODO: should we update weights regardless? (I don't think so)
    if result != expected:
      self.weights = [
          self.weights[i] + self.learning_rate * input[i] * expected
          for i in range(len(self.weights))
      ]

  def train(self,
            train_X,
            train_Y,
            log_steps=False,
            num_passes=DEFAULT_PASSES):
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
          time.sleep(0.01)

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


# take sum of x[0] and x[1]
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


functions = [function1, function2, function3, function4, function5]


def run_training(f):
  X, Y = generate_dataset(f, generator=lambda: random.uniform(-30,30))

  (X_train, X_test, Y_train,
   Y_test) = (X[:DEFAULT_SAMPLE_SIZE - DEFAULT_TEST_SIZE],
              X[DEFAULT_SAMPLE_SIZE - DEFAULT_TEST_SIZE:],
              Y[:DEFAULT_SAMPLE_SIZE - DEFAULT_TEST_SIZE],
              Y[DEFAULT_SAMPLE_SIZE - DEFAULT_TEST_SIZE:])

  # for i in range(len(X_train)):
    # print(f'X: {X_train[i]:.2f}, Y: {Y_train[i]}')
    
  p = Perceptron()

  print(f'Train acuracy before training {p.accuracy(X_train, Y_train):.2f}')
  print(f'Test accuracy before training {p.accuracy(X_test, Y_test):.2f}')

  p.train(X_train, Y_train, True)

  print(f'Train acuracy after training {p.accuracy(X_train, Y_train):.2f}')
  print(f'Test accuracy after training {p.accuracy(X_test, Y_test):.2f}')


for i in range(len(functions)):
  print(f'----\n----\nFor function{i+1}')
  run_training(functions[i])
