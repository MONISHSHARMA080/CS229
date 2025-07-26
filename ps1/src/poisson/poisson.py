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
    predictions = clf.predict(x_eval)
    np.savetxt(save_path, clf.predict(x_eval))
    # *** END CODE HERE ***
    plot_path = save_path.replace('.txt', '.jpg') if save_path.endswith('.txt') else save_path + '.jpg'
    plot_poisson_results(x_eval, y_eval, predictions, plot_path)
    
    # Print some evaluation metrics
    mse = np.mean((predictions - y_eval) ** 2)
    mae = np.mean(np.abs(predictions - y_eval))
    print(f"\n--- Poisson Regression Results ---")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Plot saved to: {plot_path}")
    print("----------------------------------\n")


def plot_poisson_results(x_eval, y_eval, predictions, save_path):
    """Plot Poisson regression results.
    
    Args:
        x_eval: Evaluation features
        y_eval: True values
        predictions: Predicted values
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 4))
    
    # Plot 1: Predicted vs Actual
    plt.subplot(1, 3, 1)
    plt.scatter(y_eval, predictions, alpha=0.6)
    min_val = min(min(y_eval), min(predictions))
    max_val = max(max(y_eval), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    plt.subplot(1, 3, 2)
    residuals = predictions - y_eval
    plt.scatter(predictions, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Feature vs Prediction (using first non-intercept feature)
    plt.subplot(1, 3, 3)
    if x_eval.shape[1] > 1:  # Has features beyond intercept
        feature_idx = 1  # First non-intercept feature
        plt.scatter(x_eval[:, feature_idx], y_eval, alpha=0.6, label='Actual', color='blue')
        plt.scatter(x_eval[:, feature_idx], predictions, alpha=0.6, label='Predicted', color='red')
        plt.xlabel(f'Feature {feature_idx}')
        plt.ylabel('Values')
        plt.title('Feature vs Values')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No features to plot', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('No Features Available')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

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
    predictions = clf.predict(x_eval)
    
    # Save predictions
    np.savetxt(save_path, predictions)
    
    # Plot results (similar to GDA plotting)
    plot_path = save_path.replace('.txt', '.jpg') if save_path.endswith('.txt') else save_path + '.jpg'
    plot_poisson_results(x_eval, y_eval, predictions, plot_path)
    
    # Print some evaluation metrics
    mse = np.mean((predictions - y_eval) ** 2)
    mae = np.mean(np.abs(predictions - y_eval))
    print(f"\n--- Poisson Regression Results ---")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Plot saved to: {plot_path}")
    print("----------------------------------\n")
    


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

    def fit(self, input:np.ndarray, output:np.ndarray):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        def gradient_vector(output, theta:np.ndarray, input:np.ndarray)->np.ndarray:
            """this function is to calculte the single gradient vector(returns a vec)
               for eg it calculates -> (y - e{theta.T * X}) X <- and returns this vector
            """
            # print(F"the shape of the theta is {theta.shape} and of the input is {input.shape} ")
            assert theta.shape[0] == input.shape[0], F"Number of features in theta and input must match for dot product., but we got the input shape to be {input.shape} and theta shape:{theta.shape}"
            output = (output - np.exp( np.dot(np.transpose(theta), input))) * input
            # print(F" the output vector of the gradient_vector() is {output}")
            return output

        def gradient_vector_sum_through_all_examples(theta:np.ndarray, input:np.ndarray, output:np.ndarray) -> np.ndarray:
            """this function goes through the entire dataset to calculate the gradient_vector"""
            gradient_sum = np.zeros(theta.shape)
            # go through all the inputs and then calculate the sum
            # wait this one will get all the input the util func got in one go and when we are iterating over them we will pass one by one (row) to gradient_vector()
            assert theta.shape != input.shape , F" the shape of theta and input must not match(for eg input(1000, 3), theta(3)), but we got the input shape to be {input.shape} and theta shape:{theta.shape}"
            for i in range(input.shape[0]):
                gradient_sum += gradient_vector(output[i], theta, input[i])
            return gradient_sum
        # loop for the learning of the theta
        initial_theta = np.zeros(input.shape[1])
        new_theta = np.zeros(input.shape[1])
        diff_btw_theta = np.inf
        print("\n\n\n\n")
        while diff_btw_theta > self.eps :
            # compute the gradient vector(total)
            gradient_vec= gradient_vector_sum_through_all_examples(initial_theta,input, output )
            assert gradient_vec.shape[0] == initial_theta.shape[0] , F"gradient_vec(shape:{gradient_vec.shape}) should be of the same shape as the theta_vec:{initial_theta.shape} as theta is of same shape of input_vec(derived from it) and gradient vec is multipied with gradient vec  "
            new_theta = initial_theta +  self.step_size * gradient_vec
            diff_btw_theta = np.linalg.norm( new_theta - initial_theta)
            print(F"in the while loop and gradient_vec shape is {gradient_vec.shape}  ---- and Initial vec shape is {initial_theta.shape} ")
            print(F" initial_theta is {initial_theta} and gradient_vec is {gradient_vec} and diff_btw_theta is {diff_btw_theta} \n")
            initial_theta = new_theta

        self.theta = initial_theta

        # *** END CODE HERE ***

    def predict(self, x:np.ndarray)->float:
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Floating-point prediction for each input, shape (n_examples,).
        """
        if not isinstance(self.theta, np.ndarray):
            raise ValueError("the theta is not of np.array type")
        # *** START CODE HERE ***

        return np.exp(np.dot(x, self.theta ))
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(lr=1e-5,
        train_path='train.csv',
        eval_path='valid.csv',
        save_path='poisson_pred.txt')
