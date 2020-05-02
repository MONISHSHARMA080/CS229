import numpy as np
import util
import sys
from random import random

sys.path.append('../linearclass')

### NOTE : You need to complete logreg implementation first!

from logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'
# Ratio of class 0 to class 1
kappa = 0.1

def main(train_path, validation_path, save_path):
    """Problem 2: Logistic regression for imbalanced labels.

    Run under the following conditions:
        1. naive logistic regression
        2. upsampling minority class

    Args:
        train_path: Path to CSV file containing training set.
        validation_path: Path to CSV file containing validation set.
        save_path: Path to save predictions.
    """
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_upsampling = save_path.replace(WILDCARD, 'upsampling')

    # *** START CODE HERE ***
    # Part (b): Vanilla logistic regression
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(validation_path, add_intercept=True)
    clf = LogisticRegression()
    clf.fit(x_train , y_train)
    probabilities = clf.predict(x_eval)
    predictions = (probabilities>0.5).astype('int')

    def classifierAccuracy(y_true, y_hat):
        return np.mean(y_true==y_hat)

    def posNegAccuracy(y_true, y_hat):
        pos = y_true==1
        posAcc = np.mean(y_true[pos]==y_hat[pos])
        negAcc = np.mean(y_true[~pos]==y_hat[~pos])
        return posAcc, negAcc

    A = classifierAccuracy(y_eval, predictions)
    posAcc, negAcc = posNegAccuracy(y_eval, predictions)
    A_bar = 0.5 * (posAcc+negAcc)
    print('Accuracy is: ', A)
    print('Positive Accuracy is: {}. Negative accuracy is {}.'.format(posAcc, negAcc))
    print('Balanced Accuracy is: ', A_bar)
    util.plot(x_eval, y_eval, clf.theta, output_path_naive.replace('txt', 'jpg'))
    # Make sure to save predicted probabilities to output_path_naive using np.savetxt()
    np.savetxt(output_path_naive, probabilities)
    # Part (d): Upsampling minority class
    # Make sure to save predicted probabilities to output_path_upsampling using np.savetxt()
    # Repeat minority examples 1 / kappa times

    def resample(x, y, kappa):
        pos = (y==1)
        x_resampled_pos = np.repeat(x[pos], 1/kappa, axis=0)
        y_resampled_pos = np.repeat(y[pos], 1 / kappa, axis=0)
        x_resampled = np.append(x_resampled_pos, x[~pos], axis=0)
        y_resampled = np.append(y_resampled_pos, y[~pos], axis=0)
        return x_resampled, y_resampled

    x_resampled, y_resampled = resample(x_train, y_train, kappa)
    clf2 = LogisticRegression()
    clf2.fit(x_resampled, y_resampled)
    probabilities2 = clf2.predict(x_eval)
    predictions2 = (probabilities2>0.5).astype('int')
    A2 = classifierAccuracy(y_eval, predictions2)
    posAcc2, negAcc2 = posNegAccuracy(y_eval, predictions2)
    A_bar2 = 0.5 * (posAcc2 + negAcc2)
    print('Accuracy is: ', A2)
    print('Positive Accuracy is: {}. Negative accuracy is {}.'.format(posAcc2, negAcc2))
    print('Balanced Accuracy is: ', A_bar2)
    util.plot(x_eval, y_eval, clf2.theta, output_path_upsampling.replace('txt', 'jpg'))
    np.savetxt(output_path_upsampling, probabilities2)
    # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='train.csv',
        validation_path='validation.csv',
        save_path='imbalanced_X_pred.txt')
