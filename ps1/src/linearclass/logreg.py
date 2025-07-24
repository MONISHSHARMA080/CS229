import numpy as np
import util



def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    print("in the main")
    X,Y =util.load_dataset(train_path)
    print(F"loaded the dataset and the X is {len(X)} and y is {Y.__len__()} ")
    LogisticRegression().fit(X,Y)

    # *** START CODE HERE ***
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
            number_of_examples =  np.shape(x)[1]
            return (-1/number_of_examples)*np.dot(np.transpose(x), y - hypothesis(theta, x))

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
            print(F"itetation {i} and the difference between thetas is {diff_between_theta} , is it <= eps {diff_between_theta <= self.eps} and the current theta is {next_theta} and prev theta is {prev_theta}\n\n")
            prev_theta =  next_theta

        print(F"the loop is over and is it <= eps {diff_between_theta <= self.eps} and diff_between_theta: {diff_between_theta}")

        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """

        # *** START CODE HERE ***
        # *** END CODE HERE ***


if __name__ == '__main__':
    try:
        main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    except Exception as e:
        print(F" there is a exception in main and it is {e} \n")
    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')
