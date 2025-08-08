import numpy as np
import util

# Noise ~ N(0, sigma^2)
sigma = 0.5
# Dimension of x
d = 500
# Theta ~ N(0, eta^2*I)
eta = 1/np.sqrt(d)
# Scaling for lambda to plot
scale_list = [0, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, ]
# List of dataset sizes
n_list = [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]

def ridge_regression(train_path, validation_path):
    """Problem 5 (d): Parsimonious double descent.
    For a specific training set, obtain theta_hat under different l2 regularization strengths
    and return validation error.

    Args:
        train_path: Path to CSV file containing training set.
        validation_path: Path to CSV file containing validation set.

    Return:
        val_err: List of validation errors for different scaling factors of lambda in scale_list.
    """
    # *** START CODE HERE ***
    # Load data sets
    train_x, train_y = util.load_dataset(train_path)
    val_x, val_y = util.load_dataset(validation_path)

    dim_of_x:int = train_x.shape[1]
    eta = 1/(np.sqrt(dim_of_x))

    lmbda = sigma ** 2 / eta **2

    # X.X^T
    xxt:np.ndarray = np.dot(train_x, train_x.T)
    identity_matrix = np.identity(xxt.shape[0])
    print(F"the identity matrix is {identity_matrix.shape} ")

    val_err= []
    for scale in scale_list:
        print(F"on the scale {scale} in the list")
        scaled_lambda = scale * lmbda
        #  XX^T + lambda*I
        regularized_matrix = xxt  + scaled_lambda * identity_matrix
        # Compute theta_hat using the formula: X^T * (XX^T + lambda*I)^(-1) * Y
        theta_pred = train_x.T @ np.linalg.pinv(regularized_matrix) @ train_y
        val_pred = val_x @ theta_pred 
        # mean square error
        err_in_pred = np.mean((val_y - val_pred)**2)
        # print(F"the val pred is {val_pred} and the val_err or error in the value is {val_err}  ")
        val_err.append(err_in_pred)
    return val_err

    # *** END CODE HERE

if __name__ == '__main__':
    val_err = []
    for n in n_list:
        val_err.append(ridge_regression(train_path='train%d.csv' % n, validation_path='validation.csv'))
    val_err = np.asarray(val_err).T
    util.plot(val_err, 'doubledescent.png', n_list)
