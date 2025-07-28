import numpy as np
import util
import sys
from random import random
from logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'
# Ratio of class 0 to class 1
kappa = 0.1

def getClassAcurracy(prediction:np.ndarray, actual_output:np.ndarray, is_positive:bool):
    """if is_positive  is true the func retuns the class acc of a1 """
    class_acc = 0
    target_class = 1 if is_positive else 0
    total_eg_in_target_class = 0
    eg_with_correct_pred = 0
    assert prediction.shape[0] == actual_output.shape[0], F" the number of predictions and the actual output array don't match"
    for i in range(prediction.shape[0]):
        assert actual_output[i] == 0 or actual_output[i] == 1, F" the actual output should be either 1 or 0, but at index:{i} it is {actual_output[i]}"
        if actual_output[i] == target_class:  
            total_eg_in_target_class += 1
            if prediction[i] == actual_output[i]:  
                eg_with_correct_pred += 1
    
        # print(F"at i:{i} the actual_output is {actual_output[i]} and target_class is {target_class} and total_eg_in_target_class is {total_eg_in_target_class} ")

    class_acc = eg_with_correct_pred / total_eg_in_target_class
    return class_acc


def compute_accuraccy(lr:LogisticRegression,validation_path:str, save_path:str):
    """computes the accuraccy, balancd_accuraccy, A0 and A1"""
    x,y=util.load_dataset(validation_path, add_intercept=False)
    res = lr.predict(x)
    total_eg = x.shape[0]
    for i in range(total_eg):
        res[i] = 1 if res[i] >= 0.5 else 0
    correct_pred = 0
    for i in range(total_eg):
        # print(F" the res[{i}] is {res[i]}")
        correct_pred += 1 if res[i] == y[i] else 0
    Accuraccy = correct_pred/ total_eg
    A0 = getClassAcurracy(res, y, False)
    A1 = getClassAcurracy(res, y, True)
    balancd_accuraccy = (1/2)* (A0 + A1)
    # print(F"A0 is {A0}, A1 is {A1}, Accuraccy is {Accuraccy} and balancd_accuraccy is {balancd_accuraccy}  ")
    return {"A0":A0, "A1":A1, "Accuraccy":Accuraccy, "balancd_accuraccy":balancd_accuraccy}


def create_balanced_dataset(x_original: np.ndarray, y_original: np.ndarray):
    """
    Creates a new dataset D0 by oversampling the minority class to balance the classes.
    The goal is to have an equal number of examples for class 0 and class 1 in the new dataset.

    Args:
        x_original: Original features from the training dataset.
        y_original: Original labels from the training dataset.

    Returns:
        Tuple[np.ndarray, np.ndarray]: x_D0, y_D0 - the balanced features and labels.
    """
    # Separate data by class
    x0 = x_original[y_original == 0] # Features for class 0
    y0 = y_original[y_original == 0] # Labels for class 0
    x1 = x_original[y_original == 1] # Features for class 1
    y1 = y_original[y_original == 1] # Labels for class 1

    n0 = x0.shape[0] # Number of majority class (class 0) examples
    n1 = x1.shape[0] # Number of minority class (class 1) examples

    x_D0 = x_original
    y_D0 = y_original

    # Determine which class is the minority and oversample it
    if n0 > n1: # Class 0 is the majority, so class 1 is the minority
        num_samples_to_add = n0 - n1 # Calculate how many samples to add to class 1
        
        if n1 > 0: # Ensure there are minority samples to pick from for oversampling
            # Randomly sample with replacement from minority class (class 1)
            minority_indices = np.random.choice(n1, num_samples_to_add, replace=True)
            x1_oversampled = x1[minority_indices]
            y1_oversampled = y1[minority_indices]

            # Concatenate original class 0 with original class 1 and oversampled class 1
            x_D0 = np.vstack((x0, x1, x1_oversampled))
            y_D0 = np.hstack((y0, y1, y1_oversampled))
        else:
            print("Warning: Minority class (class 1) has no samples in the training data. Cannot oversample.")
            # If minority class is empty, D0 will just contain the majority class
            x_D0 = x0
            y_D0 = y0
    elif n1 > n0: # Class 1 is the majority, so class 0 is the minority
        # Although the problem mentions "upsampling minority class" (implying class 1),
        # this block handles the case where class 0 might be the minority.
        num_samples_to_add = n1 - n0 # Calculate how many samples to add to class 0
        
        if n0 > 0: # Ensure there are minority samples to pick from for oversampling
            # Randomly sample with replacement from minority class (class 0)
            minority_indices = np.random.choice(n0, num_samples_to_add, replace=True)
            x0_oversampled = x0[minority_indices]
            y0_oversampled = y0[minority_indices]

            # Concatenate original class 1 with original class 0 and oversampled class 0
            x_D0 = np.vstack((x1, x0, x0_oversampled))
            y_D0 = np.hstack((y1, y0, y0_oversampled))
        else:
            print("Warning: Minority class (class 0) has no samples in the training data. Cannot oversample.")
            # If minority class is empty, D0 will just contain the majority class
            x_D0 = x1
            y_D0 = y1
    else: # The dataset is already balanced (n0 == n1)
        print("Dataset is already balanced. No oversampling needed.")
        x_D0 = x_original
        y_D0 = y_original

    # Shuffle the combined dataset to mix the original and oversampled examples
    permutation = np.random.permutation(x_D0.shape[0])
    x_D0 = x_D0[permutation]
    y_D0 = y_D0[permutation]

    return x_D0, y_D0
def main(train_path, validation_path, save_path):
    """Problem 2: Logistic regression for imbalanced labels.

    Run under the following conditions:
        1. naive logistic regression
        2. upsampling minority class

    Args:
        train_path: Path to CSV file containing training set.
        validation_path: Path to CSV file containing validation set.
        save_path: Path to save predictions.
    """
    # *** END CODE HERE
    x,y =util.load_dataset(train_path,False)
    log_reg = LogisticRegression(verbose = False)
    bal_x, bal_y = create_balanced_dataset(x,y)
    log_reg.fit(x,y)
    res=   compute_accuraccy(log_reg,validation_path,save_path)
    print(F" the result from the normal(unbalanced) dataset is {res}")

    log_reg.fit(bal_x, bal_y)
    res=   compute_accuraccy(log_reg,validation_path,save_path)
    print(F" the result from the balaned dataset is {res}")



if __name__ == '__main__':
    main(train_path='train.csv',
        validation_path='validation.csv',
        save_path='imbalanced_X_pred.txt')
