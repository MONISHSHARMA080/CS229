import numpy as np
import util



def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # *** START CODE HERE ***
    X,Y =util.load_dataset(train_path, add_intercept=True)
    print(F"loaded the dataset and the X is {len(X)} and y is {Y.__len__()} ")
    logRegression = LogisticRegression()
    logRegression.fit(X,Y)
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
    print(F"\n\n\n\n ----- loaded the valid_path dataset and theta's len is {logRegression.theta.shape}----- \n\n\n")
    util.plot(x_eval, y_eval, logRegression.theta, save_path+'.jpg')
    print(F"\n\n----plotted -----\n\n")
    logRegression.predict(x_eval)
    # Use np.savetxt to save predictions on eval set to save_path
    dataset_name = valid_path.split('/')[-1].replace('_valid.csv', '') # Extract ds1 or ds2
    logRegression.evaluate_model(logRegression, x_eval, y_eval, dataset_name)
    np.savetxt(save_path, (logRegression.predict(x_eval)>0.5).astype(int), fmt='%i')
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
        # this is the theta obtained after the training
        self.theta = None

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        

        # *** START CODE HERE ***
        # ok we need to run newton's method on this, so first of all see the formula and then try to implement this
        #things we have:-> hipothesis(), hessian(), 
        def sigmoid(predicted_value)->np.ndarray:
            return 1./(1+np.exp(-predicted_value))

        def z(theta:np.ndarray, x:np.ndarray)->np.ndarray:
            # print(F"the shape of the x  {x.shape} and theta  --{theta.shape} and their len is {len(x)}")
            assert theta.shape[0]== x.shape[1], F"In Z function the theta({theta.shape}) and X({x.shape}) should have the same shape"
            return np.dot( x, theta)
        
        
        def hypothesis(theta:np.ndarray, x:np.ndarray)->np.ndarray:
            """ hypothesis function"""
            # print(F"the shape of the x  {x.shape} and theta  --{theta.shape} in the hypothesis()")
            return sigmoid(z(theta, x))

        def loglikelyhood(theta, x, y):
            return np.sum(y*np.log(hypothesis(theta,x)) + (1-y)*np.log(1-hypothesis(theta, x)) ) 

        def firstDerivative(theta:np.ndarray, x:np.ndarray, y:np.ndarray)->np.ndarray:
            number_of_examples_for_averaging =  np.shape(x)[1]
            return (-1/number_of_examples_for_averaging)*np.dot(np.transpose(x), y - hypothesis(theta, x))

        def hessian(theta, x, y)->np.ndarray:
            n = np.shape(x)[0]
            d = np.zeros((x.shape[1], x.shape[1]))
            h = hypothesis(theta, x)
            # print(F"the shape of return value of hypothesis is {h.shape}")
            # diagonal matrix
            d = np.diag(h*(1-h))
            # print(F"the shape of hessian is {d.shape} and at H[2][2] {d[2][2]} and H[4][2] {d[4][2]} and H[5][5] {d[5][5]} ")
            return np.dot(np.transpose(x), np.dot(d,x))

        #shape/number of elements in x
        print(F"in the fit func and eps id {self.eps}")
        prev_theta = self.theta if self.theta is not None else np.zeros((x.shape[1],))
        next_theta = prev_theta.copy()
        diff_between_theta = np.inf 
        i = 0
        while diff_between_theta > self.eps :
            hessian1 = hessian(prev_theta,x, y)
            first_derivative = firstDerivative(prev_theta, x, y)

            theta_delta =   np.dot( np.linalg.inv(hessian1) , first_derivative)

            next_theta = prev_theta - theta_delta

            diff_between_theta = np.linalg.norm(next_theta - prev_theta)
            i+=1
            initial_theta = next_theta
            if self.verbose: print(F"itetation {i} and the difference between thetas is {diff_between_theta} , is it <= eps {diff_between_theta <= self.eps} and the current theta is {next_theta} and prev theta is {prev_theta}\n")
            prev_theta =  next_theta

        print(F"the loop is over and is it <= eps {diff_between_theta <= self.eps} and diff_between_theta: {diff_between_theta}\n\n\n\n\n\n\n")
        self.theta = next_theta

        # *** END CODE HERE ***

    def predict(self, x:np.ndarray)->np.ndarray:
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """

        # *** START CODE HERE ***
        def g(theta, x)->np.ndarray:
            return np.dot(x, theta)

        def sigmoid(z)->np.ndarray:
            return 1 / (1 + np.exp(-z))

        def h(theta, x)->np.ndarray:
            return sigmoid(g(theta, x)) 
        
        if self.theta is None:
            raise ValueError("the theta is none so we can't move forward")

        return  h(self.theta, x)
        # *** END CODE HERE ***

    def evaluate_model(self, model, x_eval, y_eval, dataset_name=""):
        
        """
        Evaluates the logistic regression model and prints its accuracy.

            Args:
                 model: An instance of the LogisticRegression model (already fitted).
                 x_eval: Evaluation example inputs. Shape (n_examples, dim).
                 y_eval: True labels for evaluation. Shape (n_examples,).
                dataset_name: Optional string to identify the dataset (e.g., "ds1").
        """
        # Get predicted probabilities
        y_pred_probs = model.predict(x_eval)

        # Convert probabilities to binary predictions (0 or 1)
        y_pred_binary = (y_pred_probs >= 0.5).astype(int)

        # Calculate accuracy
        accuracy = np.mean(y_pred_binary == y_eval)

        # Print the accuracy in the desired format
        print(f"The accuracy on {dataset_name} validation set using logistic regression is: {accuracy}\n\n\n")
        return accuracy

if __name__ == '__main__':
        main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')


        main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt') 


