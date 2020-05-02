import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    # Plot decision boundary on top of validation set set
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
    util.plot(x_eval, y_eval, clf.theta, save_path+'.eps')
    # Use np.savetxt to save predictions on eval set to save_path
    np.savetxt(save_path, (clf.predict(x_eval)>0.5).astype(int), fmt='%i')
    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """

        # *** START CODE HERE ***
        # theta dot x
        def g(theta, x):
            return np.dot(x, theta)

        # sigmoid
        def sigmoid(z):
            return 1 / (1 + np.exp(-z))

        # predictions of y
        def h(theta, x):
            return sigmoid(g(theta, x))

        # log-likelihood function
        def logLikelihood(theta, x, y):
            return np.sum(y * np.log(h(theta, x)) + (1 - y) * np.log(h(-theta, x)))

        # nabla log-likelihood
        def first_derivative(theta, x, y):
            return np.dot(np.transpose(x), y - h(theta, x))

        # nabla square log-likelihood
        def hessian(theta, x):
            n = x.shape[0]
            d = np.zeros((n, n))
            hx = h(theta, x)
            for i in range(n):
                d[i][i] = hx[i]
            return np.dot(np.transpose(x), np.dot(d, x))

        # Initializing theta
        if self.theta is None: self.theta = np.zeros(shape=[x.shape[1], 1])
        # Reshape y
        y = y.reshape(y.shape[0], 1)
        # Main update iteration
        for i in range(self.max_iter):
            print('Iteration ', i)
            delta_theta = np.dot(np.linalg.inv(hessian(self.theta, x)), first_derivative(self.theta, x, y))
            updateValue = np.linalg.norm(delta_theta)
            self.theta += delta_theta
            if self.verbose:
                J = -logLikelihood(self.theta, x, y)
                print('Loss of iteration {} is {}'.format(i, round(J, 5)))
            # Break if updates too small
            if updateValue < self.eps:
                break
        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """

        # *** START CODE HERE ***
        def g(theta, x):
            return np.dot(x, theta)

        def sigmoid(z):
            return 1 / (1 + np.exp(-z))

        def h(theta, x):
            return sigmoid(g(theta, x))

        return h(self.theta, x).reshape(x.shape[0])
        # *** END CODE HERE ***


if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')
