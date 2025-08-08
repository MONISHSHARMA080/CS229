# Important note: you do not have to modify this file for your homework.

import util
import numpy as np


def calc_grad(X, Y, theta):
    """Compute the gradient of the loss with respect to theta."""
    count, _ = X.shape

    probs = 1. / (1 + np.exp(-X.dot(theta)))
    grad = (Y - probs).dot(X)

    return grad

def calc_loss(X, Y, theta):
    probs = 1. / (1 + np.exp(-X.dot(theta)))
    OneMinusProbs = 1. / (1 + np.exp(X.dot(theta)))
    return np.sum(-Y * np.log(probs) - (1 - Y) * (np.log(OneMinusProbs)))

def calc_acc(X, Y, theta):
    probs = 1. / (1 + np.exp(-X.dot(theta)))
    predicts = (probs > 0.5).astype('int')
    return np.mean(predicts==Y)

def logistic_regression(X, Y, DSName:str):
    """Train a logistic regression model."""
    theta = np.zeros(X.shape[1])
    learning_rate = 0.1
    loss_values = []
    theta_norm = []
    accuracy = []
    i = 0
    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta + learning_rate * grad
        if i % 10000 == 0:
            loss_values += [calc_loss(X, Y, theta)]
            theta_norm += [np.linalg.norm(theta)]
            accuracy += [calc_acc(X, Y, theta)]
            print('Finished %d iterations' % i, F" and the weights are {theta}")
            util.plot_loss(loss_values, DSName)
            if i == 900000:
                util.plot_norm(theta_norm, DSName)
                util.plot_acc(accuracy, DSName)
                print('plotting required graphs')

        if np.linalg.norm(prev_theta - theta) < 1e-15:
            util.plot(X, Y, theta, F'plot_{DSName}.png')
            print('Converged in %d iterations' % i)

            break
    return

def main():
    print('==== Training model on data set A ====')
    Xa, Ya = util.load_csv('ds1_a.csv', add_intercept=True)
    logistic_regression(Xa, Ya, 'A')

    print('\n==== Training model on data set B ====')
    Xb, Yb = util.load_csv('ds1_b.csv', add_intercept=True)
    logistic_regression(Xb, Yb, 'B')


if __name__ == '__main__':
    main()
