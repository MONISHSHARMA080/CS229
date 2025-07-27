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
        self.degree_of_poly = 1
        self.enable_logs = enable_logs
        self.do_we_need_sin_in_phi = False

    def fit(self, X, Y, degree_of_polynomial=5, do_we_need_sin_in_phi:bool = False):
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
        #   -- the 
        #
        #
        X = self.create_poly_feature_map(degree_of_polynomial, X, do_we_need_sin_in_phi)
        print(F" created poly_feature_map X and it's shape is {X.shape} ")
        theta = self.calculate_theta_with_normal_eqn(degree_of_polynomial, X, Y, do_we_need_sin_in_phi)
        self.theta = theta
        print(F" the shape of theta after the fitting is {theta.shape} and X's shape is {X.shape} ")
        self.degree_of_poly = degree_of_polynomial
        self.do_we_need_sin_in_phi = do_we_need_sin_in_phi
        # *** END CODE HERE ***

    def calculate_theta_with_normal_eqn(self, degree_of_polynomial:int, X:np.ndarray, Y:np.ndarray, do_we_need_sin_in_phi:bool = False)->np.ndarray:
        rows_in_theta = degree_of_polynomial + 2 if  do_we_need_sin_in_phi else degree_of_polynomial + 1
        theta = np.zeros((rows_in_theta,1))
        print(F" the theta vec obtained is {theta}  and it's shape is {theta.shape} and rows in theta is {rows_in_theta} and do_we_need_sin_in_phi:{do_we_need_sin_in_phi} and degree_of_polynomial is {degree_of_polynomial}")
        assert theta.shape[0] == X.shape[1], F" theta's row should be equal to the column of the X matrix, theta:{theta.shape}, X:{X.shape}"
        assert Y.shape[0] == X.shape[0] , F" the number of rows in the y should be same as the number of rows in the x(same as no. of eg.). X:{X.shape}, y:{Y.shape} "
        XT_X = np.dot(np.transpose(X), X)

        # Calculate the right-hand side of the normal equation: (X^T * Y)
        XT_Y = np.dot(np.transpose(X), Y)

        # Solve the linear system (XT_X) * theta = XT_Y for theta using np.linalg.solve
        # This is numerically more stable than calculating the inverse.
        theta = np.linalg.solve(XT_X, XT_Y)
        # res = np.dot( inverse_of_x_dot_x, x_dot_y )
        print(F" the original theta brfore reshaping vec obtained is {theta}  and it's shape is {theta.shape}")
        if theta.ndim == 1:
            theta = theta.reshape(-1, 1)
        print(F" the theta vec obtained is {theta}  and it's shape is {theta.shape}")
        return theta

    def create_poly_feature_map(self, k, input:np.ndarray, do_we_need_sin_in_phi:bool = False)->np.ndarray:
        """
        Generates a polynomial feature map using the data x.
        The polynomial map should have powers from 0 to k
        Output should be a numpy array whose shape is (n_examples, k+1)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        number_0f_eg = input.shape[0]
        dim_of_features = k+1 if not do_we_need_sin_in_phi else k + 2
        print(F"the number_0f_eg is {number_0f_eg}, and dim_of_features {dim_of_features}")
        #final X matrix
        X = np.zeros(( number_0f_eg, dim_of_features))
        for i in range(number_0f_eg):
            # print(F"at {i}: the x[{i}] is {X[i]} and it's shape {X[i].shape}")
            X[i] = self.phi(input[i], k, do_we_need_sin_in_phi)
            # print(F"\n\n"+"-"*40+F"{i}"+"-"*40+"\n\n")
            # print(F"{X[i]}")
            # print(F"\n\n"+"-"*40+F"{i}"+"-"*40+"\n\n\n")
        print(F"the full X matrix is--> \n{X}\n and the shape of X is {X.shape} \n ")
        return X

        # *** END CODE HERE ***
    def phi(self, x:float, polynomial_degree:int, do_we_need_sin_in_phi:bool = False)->np.ndarray:
        # k+1 cause of 1 in the first place
        # print(F" what is the x: in the phi {x} ")

        #----------
        # here check if the x is a np.ndarray if it is then check if contain more than 1 element (row and column) if 
        # if does then error and if it does not and is a array then extract the first element, else if not then take the value as it is 
        #----------
        if isinstance(x, np.ndarray):
            if x.size > 1:
                 raise ValueError(f"Input array x has {x.size} elements, but only single values are supported")
            else:
             x = x.item()  # Extract the scalar value from single-element array
        last_index_of_zero_array =  polynomial_degree + 2 if do_we_need_sin_in_phi else polynomial_degree + 1
        phi_arr= np.zeros( last_index_of_zero_array)
        print(F" the shape of the feature map(phi_arr) is {phi_arr.shape}, the index of the zero array we need is {last_index_of_zero_array},  the phi_arr is {phi_arr} ")
        phi_arr[0] = 1
        # polynomial_degree + 1 casue even in the case in case of  + 2 we will do it outself
        for i in range(polynomial_degree + 1):
            phi_arr[i] = x ** i
        
        if do_we_need_sin_in_phi:
            # -1 cause in the array for eg. zero poly then the last_index_of_zero_array is 2 and the last index we can access is 1(o->1 , len =2)
            phi_arr[last_index_of_zero_array -1] = np.sin(x)
        print(F" the phi vectior we got is {phi_arr}") 
        assert phi_arr[0] == 1 , F"the first place in the phi_arr should be 1, but we found {phi_arr[0]}"
        assert phi_arr.shape[0] == last_index_of_zero_array , F" the feature map array should have a size of dim + 1 but we got {phi_arr.shape} "
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
        print(F" in the predict function ")
        if not isinstance(self.theta, np.ndarray):
            raise ValueError("the theta is not of np.array type")
        elif not isinstance(self.degree_of_poly, int):
            raise ValueError("the self.degree_of_poly is not of int type")
 
        print(F"-- in pred -- the shape of theta is {self.theta.shape} (it should be ({self.degree_of_poly+1},1)  )   and shape of X is {X.shape} ")
        X_poly = self.create_poly_feature_map(self.degree_of_poly, X, do_we_need_sin_in_phi=self.do_we_need_sin_in_phi)
        res = np.dot( X_poly, self.theta )
        # res = np.dot( self.theta, X)
        print(F" the output from predict function has a shape {res.shape} \n\n")
        return res
        # *** END CODE HERE ***



def main(train_path, small_path, eval_path, degree_of_poly = 3):
    '''
    Run all experiments
    '''
    # *** START CODE HERE ***
    # run_exp(train_path, sine=True)
    
    train_x,train_y=util.load_dataset(train_path,add_intercept=False)
    clf = LinearModel()
    clf.fit(train_x, train_y, degree_of_poly)
    print(F" the shape of the train_x is {train_x.shape} ")
    # implement the code for the graph
    # Plotting code

    plt.figure()
    plt.scatter(train_x, train_y, label='Training Data')

    # Plot the learnt hypothesis as a smooth curve
    # Create a range of x values to plot the curve
    x_range = np.linspace(min(train_x), max(train_x), 1000)
    # Predict the y values for the x_range using the fitted model
    y_pred = clf.predict(x_range)
    plt.plot(x_range, y_pred, color='red', label='Learnt Hypothesis')

    # Add labels and a legend
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Linear Regression with Degree-{degree_of_poly} Polynomial Feature Map')
    plt.legend()
    # plt.show()
    plt.savefig(F"linear_reg_with_feature_map_with_poly_degree_{degree_of_poly}.jpg")
    print(F" saved the files as:linear_reg_with_feature_map_with_poly_degree_{degree_of_poly}.jpg")
    # *** END CODE HERE ***

def compare_diff_degree_poly( train_path:str, k_value , color, do_we_need_sin_in_phi:bool = False, extra_namePfor_saved_file = None  ):
    if len(k_value) != len(color):
        raise ValueError(F" the len of k value is not same as the no. of color , k_val:{len(k_value)}, color:{len(color)} ")
    
    train_x, train_y = util.load_dataset(train_path, add_intercept=False)

    plt.figure(figsize=(10, 6))
    plt.scatter(train_x, train_y, label='Training Data', )

    x_range = np.linspace(min(train_x), max(train_x), 1000).reshape(-1, 1)
    saved_img_name = ""

    for k, color in zip(k_value, color):
        clf = LinearModel()
        print(F" in the {k} degree of poly")
        clf.fit(train_x, train_y, degree_of_polynomial=k, do_we_need_sin_in_phi= do_we_need_sin_in_phi)
        y_pred = clf.predict(x_range)
        plt.plot(x_range, y_pred, color=color, label=f'k={k} Hypothesis')
        saved_img_name+= "-"+str(k)

    saved_img_name += str(extra_namePfor_saved_file)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Polynomial Regression with Different Degrees')
    plt.legend()

    plt.savefig(f'comparison{saved_img_name}.jpg')
    print(f"\n Saved plot as comparison{saved_img_name}.jpg \n")


if __name__ == '__main__':
    # main(train_path='train.csv',
    #     small_path='small.csv',
    #     eval_path='test.csv', degree_of_poly=3)
    # compare_diff_degree_poly(train_path="train.csv", k_value=[0,1,2,3,5,10,20], color=['blue', 'green', 'red', 'purple', 'yellow', 'violet', 'black'], do_we_need_sin_in_phi=True, extra_namePfor_saved_file="_on_test" )
    compare_diff_degree_poly(train_path="small.csv", k_value=[1,2,5,10,20], color=['blue', 'green', 'red', 'purple', 'yellow'], do_we_need_sin_in_phi=False, extra_namePfor_saved_file="_on_small_dtst" )
    # compare_diff_degree_poly(train_path="train.csv", k_value=[3,5,10,20], color=['blue', 'green', 'red', 'purple'], do_we_need_sin_in_phi=True )
