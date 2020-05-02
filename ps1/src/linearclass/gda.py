import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    # *** START CODE HERE ***
    # Train a GDA classifier
    clf = GDA()
    clf.fit(x_train, y_train)
    # Plot decision boundary on validation set
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
    util.plot(x_eval, y_eval, clf.theta, save_path+'.eps')
    # Use np.savetxt to save outputs from validation set to save_path
    np.savetxt(save_path, (clf.predict(x_eval) > 0.5).astype(int))
    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
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
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        def quadratic(x, A):
            return np.dot(np.dot(x.T,A),x)
        y = y.reshape([y.shape[0], 1])
        # Find phi, mu_0, mu_1, and sigma
        phi = np.mean(y)
        mu_0 = np.transpose(np.sum(x * (y==0).astype(int), keepdims=True, axis=0)/np.sum((y==0))) # (d, 1)
        mu_1 = np.transpose(np.sum(x * y, keepdims=True, axis=0)/np.sum(y)) # (d, 1)
        sum = np.zeros((x.shape[1],x.shape[1]))
        n_feature = x.shape[1]
        for i in range(x.shape[0]):
            if y[i]: mu = mu_1
            else: mu = mu_0
            vol_vector = x[i].reshape([n_feature, 1]) - mu
            sum += np.dot(vol_vector, np.transpose(vol_vector))
        sigma = 1 / x.shape[0] * sum
        inv_sigma = np.linalg.inv(sigma)
        # Write theta in terms of the parameters
        theta_1 = np.transpose(np.dot(np.transpose(mu_1-mu_0), inv_sigma)).reshape(2)
        theta_0 = 1/2 * (quadratic(mu_0, inv_sigma) - quadratic(mu_1, inv_sigma)) + np.log((1 - phi)/phi)
        # Store theta in such a manner: (theta_0, theta_1[0], theta_1[1]).T
        self.theta = np.zeros((theta_1.shape[0]+1,1))
        self.theta[0,0] = theta_0
        self.theta[1:,0] = theta_1
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        def g(theta, x):
            return np.dot(x, theta)

        def sigmoid(z):
            return 1/(1+np.exp(-z))

        def h(theta, x):
            return sigmoid(g(theta, x))

        predicts = h(self.theta, x)
        return predicts
        # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')
