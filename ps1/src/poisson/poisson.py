import numpy as np
import util
import matplotlib.pyplot as plt

def main(lr, train_path, eval_path, save_path):
    """Problem: Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        save_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    clf = PoissonRegression(step_size=lr)
    clf.fit(x_train, y_train)
    # Run on the validation set, and use np.savetxt to save outputs to save_path
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    np.savetxt(save_path, clf.predict(x_eval))
    # *** END CODE HERE ***


class PoissonRegression:
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=1e-5, max_iter=10000000, eps=1e-5,
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
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # eta
        def g(theta, x):
            return np.dot(x, theta)

        # predictions of y
        def h(theta, x):
            return np.exp(g(theta, x))

        fac = lambda x: np.math.factorial(x)

        # loss function
        def loss(theta, x, y):
            eta = g(theta, x)
            lambd = np.exp(eta)
            factorials = np.array(list(map(fac, y))).reshape([y.shape[0],1]).astype('float')
            return np.mean(np.log(factorials) + lambd - eta * y)

        # l2 loss
        def l2loss(theta, x, y):
            y_hat = h(theta, x)
            return np.mean(0.5 * (y - y_hat)**2)

        (n,dim) = x.shape
        if self.theta is None: self.theta = np.zeros((dim,1))
        # reshape y
        y = y.reshape([y.shape[0],1])
        for i in range(self.max_iter):
            lambd = h(self.theta, x)
            updateVector = self.step_size * (np.dot(x.transpose(), (y - lambd)))
            updateVector = self.step_size * (np.dot(x.transpose(), (y - lambd)))
            self.theta += updateVector
            updateValue = np.linalg.norm(updateVector)
            J = loss(self.theta, x, y)
            if self.verbose:
                print('loss of iteration {} is {}'.format(i, round(J, 5)))
            if updateValue < self.eps: break
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Floating-point prediction for each input, shape (n_examples,).
        """
        # *** START CODE HERE ***
        def g(theta, x):
            return np.dot(x, theta)

        # predictions of y
        def h(theta, x):
            return np.exp(g(theta, x))

        return h(self.theta, x)
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(lr=1e-5,
        train_path='train.csv',
        eval_path='valid.csv',
        save_path='poisson_pred.txt')
