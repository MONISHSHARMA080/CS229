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
    gdaClassifier = GDA()
    gdaClassifier.fit(x_train, y_train)
    # Plot decision boundary on validation set
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
    util.plot(x_eval, y_eval, gdaClassifier.theta, save_path+'.jpg')
    # Use np.savetxt to save outputs from validation set to save_path
    np.savetxt(save_path, (gdaClassifier.predict(x_eval) > 0.5).astype(int))
    # -- quantify the model prediction
    predictions = (gdaClassifier.predict(x_eval) >= 0.5).astype(int)
    evaluate_model(predictions, y_eval, save_path)
    # *** END CODE HERE ***

def evaluate_model(predictions, actual, save_path):
    """
    Evaluates the performance of the model by comparing predictions with actual labels.

    Args:
        predictions: The predicted labels from the model.
        actual: The actual labels from the validation set.
        save_path: Path to the file where predictions are saved.
    """
    # Load the predictions from the saved file to ensure we're using the same data
    saved_predictions = np.loadtxt(save_path)

    if saved_predictions.shape != actual.shape:
        print("Warning: Shape of predictions and actual labels do not match.")
        return

    num_examples = len(actual)
    correct_predictions = np.sum(saved_predictions == actual)
    accuracy = correct_predictions / num_examples

    print("\n--- Model Performance Evaluation ---")
    print(f"Number of examples: {num_examples}")
    print(f"Number of correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("------------------------------------\n")

    # print("Detailed comparison of predictions and actual labels:")
    # for i in range(num_examples):
    #     result = "Success" if saved_predictions[i] == actual[i] else "Failure"
    #     print(f"Example {i+1}: Actual = {actual[i]}, Predicted = {saved_predictions[i]}, Result = {result}")

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
        self.theta_0 = theta_0
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x:np.ndarray, y:np.ndarray):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        def calculate_mu(is_this_one:bool, output:np.ndarray, input:np.ndarray)->np.ndarray:
            # this is the value we compare the output with, if output is like the val then we add 1 in it's place
            val =0
            if is_this_one: 
                val = 1
            assert input.shape[0] == output.shape[0], F"there should be same number of input as there are output, but we found inputs:{input.shape[0]} and outputs:{output.shape[0]}"
            print(F"the y[1] is {output[1]}")
            numerator = np.zeros(input.shape[1])
            denominator = 0
            for i in range(input.shape[0]):
                y_val = 1 if val == output[i] else 0
                numerator += y_val * x[i]
                denominator += y_val
                # print(F"i:{i} the new numerator:{numerator} and denominator:{denominator} and mu(num/deno) is {numerator/ denominator} ")
            
            return numerator/ denominator
            
        def phi(output: np.ndarray) -> float:
            """ phi is the probability of y being true """
            print(f"the shape of output is {output.shape}")
            
            output_val = 0
            for i in range(output.shape[0]):
                output_val += 1 if output[i] == 1 else 0  # Fixed: check output[i], not i
                # print(f"the output(phi in process) is {output_val}")
            
            output_val = output_val * (1/output.shape[0])
            print(f"\n ---- the phi calculated is {output_val}\n")
            return output_val


    
        def covariance_matrix_or_sigma(input:np.ndarray, mu0:np.ndarray, mu1 :np.ndarray, output:np.ndarray)->np.ndarray:
            rows_in_x, dimension_of_features = input.shape
            n = input.shape[0] # number of inputs
            sigma = np.zeros((dimension_of_features, dimension_of_features))
            print(F"the shape of the sigma matrix is {sigma.shape} ")
            for i in range(n):
                if output[i] == 0:
                    diff = input[i] - mu0
                else:
                    # output is 1
                    diff = input[i] - mu1
                # dot product produces a scaler when we don't have a matrix (just a vec, i.e. [] and matrix X matrix is a vector ), so for the 1D we will use outer
                sigma += np.outer(diff, diff)
                # print(F" at:{i} the sigma matrix is {sigma}")
            print(F"\n\n---- the final sigma matrix is {(1/n)*sigma}-----\n\n")
            print(F"the shape of the sigma matrix is {sigma.shape} ")
            return 1/n*sigma
        def quadratic_form(mew:np.ndarray, sigmma:np.ndarray):
            return np.dot(np.dot(np.transpose(mew),sigmma), mew)
        
        # now calculate the  theta and theta(knot) and boom we are ready

        phi = phi(y)
        mu0 = calculate_mu(False, y, x)
        mu1 = calculate_mu(True, y, x)
        sigma = covariance_matrix_or_sigma(x, mu0, mu1,y)
        
        inverse_sigma = np.linalg.inv(sigma)
        print(F"inverse sigma calculated and it is:{inverse_sigma} and the shape of the quadratic_form is { quadratic_form(mu1, inverse_sigma).shape}")
        
        theta_0:np.ndarray = (-1/2) * ( quadratic_form(mu1, inverse_sigma) - quadratic_form(mu0, inverse_sigma) ) + np.log(phi/(1-phi))
        theta:np.ndarray = np.dot( inverse_sigma, (mu1 - mu0))
        print(F"\n\n the theta knot is {theta_0} --and theta is {theta} \n\n")
        # self.theta = theta
        # self.theta_0 = theta_0
        # -- we are storing both of them in a single vector for ease of use --
        print(F" shape of theta_0 is:{theta_0.shape} and it is {theta_0} and shape of theta is {theta.shape} and it is {theta} ")
        
        self.theta = np.zeros((theta.shape[0] +1, 1)) # a vertical vec that has same number of rows as the theta did +1 for the theta_0(scaler)
        self.theta[0,0] = theta_0
        self.theta[1:,0] = theta
        print(F" the new vector/single theta is {self.theta} and it's shape is {self.theta.shape} ")
        print(F"\n\n--------- training completed --------\n\n")
        # print(F" predicting at index {1}-> {self.predict(x[1])} and is it == y[{1}] -> {self.predict(x[1]) == y[1] }  ")

        # *** END CODE HERE ***

    def predict(self, x:np.ndarray)->float:
        """Make a prediction given new inputs x.
        Args:
            x: Inputs of shape (n_examples, dim).
        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return 1/(1+np.exp( -( np.dot( x, self.theta)  )  ))
        # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')


