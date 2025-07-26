import util
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

np.seterr(all='raise')


factor = 2.0

class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None, enable_logs:bool=False):
        """
        Args:
            theta: Weights vector for the model.
        """
        self.theta = theta
        self.enable_logs = enable_logs

    def fit(self, X, Y, degree_of_polynomial=3):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the normal equations.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        #
        # let's walk through what we are going to do
        #
        # --- in linear reg. with normal eqn, we calculate the theta in one shot without any batch/schoastic gradient descent(see notes for formula)
        #
        # --- how do we do it(get X): (as seen in notes) the X is a nxd matrix, where n is number of eg and d is number of features
        # --- how do we do it(get Y): (as seen in notes) the Y is a nx1 matrix, where n is number of eg.
        #
        # -- but in the feature map we do the theta*phi(x) 
        # -- what is phi(x):->  [1,x,x^2,...x^k]^T        --(if x has more than 1 dim then it will be x1, x2, x1*x2, x1^2....)
        #
        # -- how to calculate the X matrix:  loop through every x(n times), and at every x(at i let's say) calculate the phi(xi)
        # --                                 and put that in the X vector's (nxd) X[i] and we have a x vector
        #
        # eg. | 1, x(1), x^2(1), x^3(1) | <-- phi(x(1))
        #    | 1, x(2), x^2(2), x^3(2) |
        #   | 1, x(3), x^2(3), x^3(3) |  <-- phi(x(3))
        #
        #   -- from here on it is simple linear algebra calculation
        #
        #
        X = self.create_poly_feature_map(degree_of_polynomial, X)
        theta = self.calculate_theta_with_normal_eqn(degree_of_polynomial, X, Y)
        self.theta = theta
        # *** END CODE HERE ***

    def calculate_theta_with_normal_eqn(self, degree_of_polynomial:int, X:np.ndarray, Y:np.ndarray)->np.ndarray:
        theta = np.zeros((degree_of_polynomial + 1,))
        assert theta.shape[0] == X.shape[1], F" theta's row should be equal to the column of the X matrix, theta:{theta.shape}, X:{X.shape}"
        assert Y.shape[0] == X.shape[0] , F" the number of rows in the y should be same as the number of rows in the x(same as no. of eg.). X:{X.shape}, y:{Y.shape} "
        inverse_of_x_dot_x = np.linalg.inv( np.dot( np.transpose(X), X )  )
        x_dot_y = np.dot( np.transpose(X), Y)
        res = np.dot( inverse_of_x_dot_x, x_dot_y )
        print(F" the theta vec obtained is {res}")
        return res

    def create_poly_feature_map(self, k, input:np.ndarray)->np.ndarray:
        """
        Generates a polynomial feature map using the data x.
        The polynomial map should have powers from 0 to k
        Output should be a numpy array whose shape is (n_examples, k+1)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        number_0f_eg = input.shape[0]
        dim_of_features = k+1
        print(F"the number_0f_eg is {number_0f_eg}, and dim_of_features {dim_of_features}")
        #final X matrix
        X = np.zeros(( number_0f_eg, dim_of_features))
        for i in range(number_0f_eg):
            X[i] = self.phi(input[i], k)
            print(F"\n\n"+"-"*40+F"{i}"+"-"*40+"\n\n")
            print(F"{X[i]}")
            print(F"\n\n"+"-"*40+F"{i}"+"-"*40+"\n\n\n")
        print(F"the full X matrix is--> \n{X}\n\n ")
        return X

        # *** END CODE HERE ***
    def phi(self, x:float, polynomial_degree:int)->np.ndarray:
        # k+1 cause of 1 in the first place
        # print(F" what is the x in the phi {x} ")

        phi_arr= np.zeros( polynomial_degree+1 )
        phi_arr[0] = 1
        for i in range(polynomial_degree + 1):
            phi_arr[i] = x ** i
        
        print(F" the phi vectior we got is {phi_arr}") 
        assert phi_arr[0] != 0 , F"the first place in the phi_arr should be 1, but we found {phi_arr[0]}"
        assert phi_arr.shape[0] == polynomial_degree +1 , F" the feature map array should have a size of dim + 1 but we got {phi_arr.shape} "
        return phi_arr


    def predict(self, X:np.ndarray)->float:
        """
        Make a prediction given new inputs x.
        Returns the numpy array of the predictions.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        res = np.dot(X, self.theta)
        print(F" the output from predict function is {res} and it's shape is {res.shape} \n\n")
        return res
        # *** END CODE HERE ***



def main(train_path, small_path, eval_path):
    '''
    Run all experiments
    '''
    # *** START CODE HERE ***
    # run_exp(train_path, sine=True)
    
    train_x,train_y=util.load_dataset(train_path,add_intercept=False)
    clf = LinearModel()
    clf.fit(train_x, train_y)
    print(F" the shape of the train_x is {train_x.shape} ")
    # implement the code for the graph
    # Plotting code
    plt.figure()
    plt.scatter(train_x, train_y, label='Training Data')
    
    # Generate points for the smooth curve
    x_smooth = np.linspace(min(train_x), max(train_x), 100).reshape(-1, 1)
    x_smooth_poly = clf.create_poly_feature_map(3, x_smooth)
    y_smooth = clf.predict(x_smooth_poly)
    
    plt.plot(x_smooth, y_smooth, 'r-', label='Learnt Hypothesis (degree 3)')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Regression with Degree-3 Polynomial Feature Map')
    plt.legend()
    plt.grid(True)
    plt.savefig('polynomial_regression_plot.jpg')
    # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='train.csv',
        small_path='small.csv',
        eval_path='test.csv')
