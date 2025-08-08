import collections

import numpy as np

import util
import svm



def get_words(message:str)-> list[str]:
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    lower_message =  message.lower()
    punctuation = ".,!?;:()*[]{}\\-/\\"
    translator = str.maketrans(punctuation, " " * len(punctuation))
    message = lower_message.translate(translator)
    return message.split()
    
    # *** END CODE HERE ***


def create_dictionary(messages:list[str]) -> dict[str, int] :
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***
    # use to count how many times the word is in the dict
    word_dict: dict[str, int] = {}
    for sentence in messages:
        words = get_words(sentence)
        for word in words:
            if word not in word_dict:
                word_dict[word] = 1
            else:
                word_dict[word] += 1  

    print(F"got the word and checking the dict size -> {word_dict.__len__()}")
    # useful is >=5 
    word_dict_with_useful_words: dict[str, int] = {}
    index=0
    for word in word_dict:
        # print(F" the word in word_dict is {word}, and occurance is  {word_dict[word]}")
        if  word_dict[word] >= 5:
            if word not in word_dict_with_useful_words:
                word_dict_with_useful_words[word] = index
            else:
                word_dict_with_useful_words[word] += 1
            # print(F"at {index}, {word}:{word_dict[word]}(occurred) and in useful dict it is on index {word_dict_with_useful_words[word]} ")
            index= index + 1

    print(F"got the word and checking the size of imp word dict-> {word_dict_with_useful_words.__len__()}")
    return word_dict_with_useful_words
    # *** END CODE HERE ***


def transform_text(messages:list[str], word_dictionary:dict[str, int])->np.ndarray:
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    of the vocabulary appears in each message. 
    Each row in the resulting array should correspond to each message 
    and each column should correspond to a word of the vocabulary.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
        Where the component (i,j) is the number of occurrences of the
        j-th vocabulary word in the i-th message.
    """
    # *** START CODE HERE ***
    text_matrix = np.zeros((len(messages), len(word_dictionary)))
    for  i,val in enumerate(messages):
        sentence = get_words(val)
        for word in sentence:
            if word in word_dictionary:
                col_or_word_index = word_dictionary[word]
                # print(F"row:{i} the word:{word} in word_dictionary at {col_or_word_index} and len of word_dictionary: {word_dictionary.__len__()} ")
                text_matrix[i, col_or_word_index] +=1
    
    print(F" the text_matrix is {text_matrix.shape} ")
    return text_matrix
    # *** END CODE HERE ***


def fit_naive_bayes_model(matrix:np.ndarray, labels:np.ndarray):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """
    # *** START CODE HERE ***
    assert isinstance(matrix, np.ndarray) , F"the matrix(arg) is not a instance of np.ndarray, type mismatch , we got {type(matrix)}"
    assert isinstance(labels, np.ndarray ) , F"the labels(arg) is not a instance of np.ndarray, type mismatch, we got {type(labels)} "
    print(F"the label is {labels[33]} and shape is {labels.shape}")
    num_documents , vocab_size = matrix.shape
    assert num_documents == len(labels), F"the num of emails/text({num_documents}) should be equal to the number of lables({len(labels)})"
    num_spam_docs = np.sum(labels == 1) # since it consist of 1,0 , adding it will give me total spam
    num_non_spam_docs = num_documents - num_spam_docs  
    p_y_true:float = num_spam_docs/num_documents
    p_y_false = 1 - p_y_true
    log_p_y_true:float = np.log(p_y_true)
    log_p_y_false:float = np.log(p_y_false)
    print(F"the prob of y is {p_y_true}")

    # res_arr = np.zeros((vocab_size,1)) # for numerator of phi
    # --- if j word is present in the i'th eg then  and the y is = class(1|0) then we add 1 to the array but the for loop in python is slow so we 
    # ( 2 res array one for the Y = 1 and another for the Y = 0 )
    #
    # if the row i is spam(based on y) then it will keep it and discard the other, so we have a array that match the comp 
    #
    # -- note this is a multimodal event model i.e the a single word can appear many times and so the word in matrix [i][j] might not be binary as it appears
    # many times
    spam_doc_array = matrix[labels == 1]
    total_spam_word_counts:float = np.sum(spam_doc_array)

    non_spam_doc_array = matrix[labels == 0]
    total_non_spam_word_counts:float = np.sum(non_spam_doc_array )

    print(F" the spam_doc shape is {spam_doc_array.shape} and the non spam_doc is  {non_spam_doc_array.shape} ")
    print(F"the total_spam_word_counts count is {total_spam_word_counts} and total_non_spam_word_counts is {total_non_spam_word_counts}  ")


    # -- calc for the spam array -- or phi_pos
    # now we need a array that tells us the number of time jth word word[i][j] comes in spam/non-spam 

    sum_word_counts_in_spam:np.ndarray = np.sum(spam_doc_array, axis=0)

    numerator = 1 + sum_word_counts_in_spam
    denominator = total_spam_word_counts + vocab_size # vocab_size for the multinomial laplace smoothing
    
    phi_pos:np.ndarray = numerator/denominator
    


    # --- calc for the phi neg.

    sum_word_counts_in_not_spam:np.ndarray = np.sum(non_spam_doc_array, axis=0)
    numerator = 1 + sum_word_counts_in_not_spam
    denom =  vocab_size + total_non_spam_word_counts
    phi_neg:np.ndarray = numerator/denom

    print(F" type of log_prob_y_spam:{type(np.log(p_y_true))} log_phi_pos: {type(np.log(phi_pos))}  ")
    return {
        'log_prob_y_spam': np.log(p_y_true),
        'log_prob_y_not_spam': np.log(p_y_false),
        'log_phi_pos': np.log(phi_pos),
        'log_phi_neg': np.log(phi_neg),
        'phi_pos': phi_pos,
        'phi_neg': phi_neg,
    }


    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model, matrix:np.ndarray)->np.ndarray:
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containing the predictions from the model
    """
    log_prob_y_spam = model['log_prob_y_spam']
    log_prob_y_not_spam = model['log_prob_y_not_spam']
    log_phi_pos = model['log_phi_pos']
    log_phi_neg = model['log_phi_neg']
    assert isinstance(log_prob_y_spam, float ), F"log_prob_y_spam is not of type(expected) float, it is {type(log_prob_y_spam)}"
    assert isinstance(log_prob_y_not_spam, float ), F"log_prob_y_not_spam is not of type(expected) float, it is {type(log_prob_y_not_spam)}"
    assert isinstance(log_phi_pos, np.ndarray ), F"log_phi_pos is not of type(expected) np.ndarray, it is {type(log_phi_pos)}"
    assert isinstance(log_phi_neg, np.ndarray ), F"log_phi_neg is not of type(expected) np.ndarray, it is {type(log_phi_neg)}"

    predcition_spam = np.sum ( log_phi_pos * matrix, axis=1, keepdims=True      ) + log_prob_y_spam
    predcition_not_spam =  np.sum(log_phi_neg * matrix, axis=1, keepdims=True) +log_prob_y_not_spam 

    print(F" the predcition_spam's shape is {predcition_spam.shape} and predcition_not_spam shape is {predcition_not_spam.shape} \n")
    # print(F" the predcition_spam is {predcition_spam} and predcition_not_spam  {predcition_not_spam} ")
    predicts = np.zeros((matrix.shape[0],1))
    predicts[predcition_spam > predcition_not_spam] = 1
    # print(F" the prediction[] {predicts.shape} and {predicts} ")

    return predicts
    
    # *** START CODE HERE ***
    # *** END CODE HERE ***


def get_top_five_naive_bayes_words(model, dictionary:dict[str, int])->list[str]:
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in part-c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: A list of the top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***
    log_prob_y_spam = model['log_prob_y_spam']
    log_prob_y_not_spam = model['log_prob_y_not_spam']
    log_phi_pos = model['log_phi_pos']
    log_phi_neg = model['log_phi_neg']
    phi_pos = model['phi_pos']
    phi_neg = model['phi_neg']

    assert isinstance(log_prob_y_spam, float ), F"log_prob_y_spam is not of type(expected) float, it is {type(log_prob_y_spam)}"
    assert isinstance(log_prob_y_not_spam, float ), F"log_prob_y_not_spam is not of type(expected) float, it is {type(log_prob_y_not_spam)}"
    assert isinstance(log_phi_pos, np.ndarray ), F"log_phi_pos is not of type(expected) np.ndarray, it is {type(log_phi_pos)}"
    assert isinstance(log_phi_neg, np.ndarray ), F"log_phi_neg is not of type(expected) np.ndarray, it is {type(log_phi_neg)}"
    assert isinstance(phi_neg, np.ndarray ), F"phi_neg is not of type(expected) np.ndarray, it is {type(phi_neg)}"
    assert isinstance(phi_pos, np.ndarray ), F"phi_pos is not of type(expected) np.ndarray, it is {type(phi_pos)}"

    original_score = (log_phi_pos - log_phi_neg)
    print("")
    print(F" original_score is {original_score[40:100]} ")
    # We sort the negative scores to get indices in descending order of the original scores
    number_of_most_likely_word = 5
    # Find the indices of the top 5 most indicative words.
    # np.argsort() returns the indices that would sort the array in ascending order.
    # We want the highest scores, so we take the last 5 elements of the argsort result.
    # The [::-1] slice then reverses this to get the indices in descending order of score.
    # This is a concise and efficient way to get the top N indices.
    index_of_most_likely_word =np.argsort(original_score, axis=0)[-number_of_most_likely_word:][::-1]
    print(F"-- {index_of_most_likely_word}  --")

    top_five_words:list[str]= []
    for i in range(number_of_most_likely_word):
        # here i repres the index in the res array as it is sensitive
        for word_key,index_of_word in dictionary.items():
            # print(F" the key is {word_key} and item/index is {index_of_word}")
            if dictionary[word_key] == index_of_most_likely_word[i]:
                # print(F"the key of word is {word_key} and index of most likely word {index_of_most_likely_word[i]} ")
                # print(F"the word {word_key}  at index {dictionary[word_key]} and the index in index_of_most_likely_word is {i} and it is {index_of_most_likely_word[i]} ")
                top_five_words.append(word_key)
    print(f'the top 5 words are {top_five_words}')
    return top_five_words

    # *** END CODE HERE ***


def compute_best_svm_radius(train_matrix:np.ndarray, train_labels:np.ndarray, val_matrix:np.ndarray, val_labels:np.ndarray, radius_to_consider:list):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spam or not spam labels for the training data
        val_matrix: The word counts for the validation data
        val_labels: The spam or not spam labels for the validation data
        radius_to_consider: The radius values to consider

    Returns:
        The best radius which maximizes SVM accuracy.
    """
    # *** START CODE HERE ***
    assert isinstance(train_labels, np.ndarray)
    assert isinstance(train_matrix, np.ndarray)
    assert isinstance(val_matrix, np.ndarray)
    assert isinstance(val_labels, np.ndarray)
    assert isinstance(radius_to_consider, list)

    accuracy_list = [0.0]*len(radius_to_consider)
    for i,radius  in enumerate(radius_to_consider):
        precition = svm.train_and_predict_svm(train_matrix, train_labels, val_matrix, radius)
        accuracy_list[i] =np.mean(precition==val_labels)

    print(F" the accuracy_list is {accuracy_list}")
    best_fit = 0
    best_radius_index =0
    for i,j in enumerate(accuracy_list):
        if j >= best_fit:
            best_fit = j
            best_radius_index =  i

    print(F" the best fit for radius is at {best_radius_index}  and it is {best_fit} and the best fit radius is {radius_to_consider[best_radius_index]}")
    return radius_to_consider[best_radius_index]
    
    # *** END CODE HERE ***


def main():
    train_messages, train_labels = util.load_spam_dataset('spam_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('spam_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('spam_test.tsv')

    dictionary = create_dictionary(train_messages)

    print('Size of dictionary: ', len(dictionary))

    util.write_json('spam_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)
    print(F" the train matrix is {train_matrix.shape}")

    np.savetxt('spam_sample_train_matrix', train_matrix[:100,:])

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt('spam_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    util.write_json('spam_top_indicative_words', top_5_words)

    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])

    util.write_json('spam_optimal_radius', optimal_radius)

    print('The optimal SVM radius was {}'.format(optimal_radius))

    svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)

    svm_accuracy = np.mean(svm_predictions == test_labels)

    print('The SVM model had an accuracy of {} on the testing set'.format(svm_accuracy, optimal_radius))


if __name__ == "__main__":
    main()
