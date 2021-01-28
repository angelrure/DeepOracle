"""
Utility module for the oracle and mutant analysis modules.
"""

import pandas as pd

def undo_dummy(sparse_matrix, label_encoder):
    """
    Transforms a dummy dataframe into the original labels.
    Params:
        spare_matrix (numpy.sparse_matrix): the encoded data.
        label_encoder (sklearn.preprocessing.LabelEncoder): the label encoder.
    Returns:
        data (list): the original labels.
    """
    data = pd.DataFrame(sparse_matrix.toarray()) 
    data = data.idxmax(1)
    data = label_encoder.inverse_transform(data)
    return data

def calculate_accuracy(oracle_output, sut_output):
    """
    Function to calculate the accuracy of the oracle model.
    Params:
        oracle_output(list): the labels predicted by the model.
        sut_output(list): the real labels.
    Returns:
        accuracy (float): returns the accuracy as the proportion of 
                          predictions that matches the real label.
    """
    accuracy = (oracle_output == real_test).sum()/real_test.shape[0]
    print(f'Models accuracy: {accuracy*100}%')
    return accuracy

def true_positives(oracle_output, sut_output, mutant_output):
    """
    Function to calculate the true positives of a mutant analysis.
    THey are the predictions that match the output and the mutant.
    Params:
        oracle_output(list): the labels predicted by the model.
        sut_output(list): the real labels.
        mutant_output(list): what the mutated program predicts.
    Returns:
        TP (pd.Series): whether each element is a true positive. 
        TP_n (int): the number of true positives.
    """
    TP = (oracle_output==sut_output) & (mutant_output==sut_output)
    TP_n = TP.sum()/len(TP)
    return TP, TP_n

def true_negative(oracle_output, sut_output, mutant_output):
    """
    Function to calculate the true negatives of a mutant analysis.
    THey are the predictions that match the output but not the mutant.
    Params:
        oracle_output(list): the labels predicted by the model.
        sut_output(list): the real labels.
        mutant_output(list): what the mutated program predicts.
    Returns:
        TN (pd.Series): whether each element is a true negative. 
        TN_n (int): the number of true negatives.
    """
    TN  = (oracle_output==sut_output) & (~(mutant_output==sut_output))
    TN_n = TN.sum()/len(TN)
    return TN, TN_n

def false_positives(oracle_output, sut_output, mutant_output):
    """
    Function to calculate the false positives of a mutant analysis.
    THey are the predictions that do not match the output but the
    match the mutant.
    Params:
        oracle_output(list): the labels predicted by the model.
        sut_output(list): the real labels.
        mutant_output(list): what the mutated program predicts.
    Returns:
        FP (pd.Series): whether each element is a false positive.
        FP_n (int): the number of false positives.
    """
    FP  = (oracle_output!=sut_output) & (mutant_output==oracle_output)
    FP_n = FP.sum()/len(FP)
    return FP, FP_n

def false_negatives(oracle_output, sut_output, mutant_output):
    """
    Function to calculate the false negatives of a mutant analysis.
    THey are the predictions that do not match the output and the
    mutant matches the sut.
    Params:
        oracle_output(list): the labels predicted by the model.
        sut_output(list): the real labels.
        mutant_output(list): what the mutated program predicts.
    Returns:
        FN (pd.Series): whether each element is a false negative.
        FN_n (int): the number of false negatives.
    """
    # THe oracle is incorrect
    FN = (oracle_output!=sut_output) & (mutant_output==sut_output)
    FN_n = FN.sum()/len(FN)
    return FN, FN_n

def run_four_analysis(oracle_output, sut_output, mutant_output):
    """
    Computes the true positives, true negatives, false positives and
    false negatives for a mutant anlaysis. The definitions are as follows:
    1. True Positive: All the results are the same. Therefore, the comparator reports “No fault”. 
    This means there is actually neither any fault in the mutated version nor the oracle. 
    True positive represents the successful test cases.
    2. True Negative: Although the expected and the oracle results are the same, they are different 
    from the mutated results. In this case, the oracle results are correct. Therefore, the oracle 
    correctly finds a fault in the mutated version.
    3. False Positive: Both the oracle and mutated version produced the same incorrect results. 
    Therefore, they are different from the expected results and faults in the oracle and the mutated 
    version are reported. To put it differently, the oracle missed a fault.
    4. False Negative: The mutated and the expected results are the same, but they are different 
    from the oracle results. Thus, the comparator reports a faulty oracle.

    Params:
        oracle_output(list): the labels predicted by the model.
        sut_output(list): the real labels.
        mutant_output(list): what the mutated program predicts.

    Returns:
        dict: a dictionary containing all the results from the anlaysis.
    """
    TP, TP_n = true_positives(oracle_output, sut_output, mutant_output)
    TN, TN_n = true_negative(oracle_output, sut_output, mutant_output)
    FP, FP_n = false_positives(oracle_output, sut_output, mutant_output)
    FN, FN_n = false_negatives(oracle_output, sut_output, mutant_output)
    return ({'True Negatives':TN, 'True Positives':TP, 'False Positives':FP, 'False Negatives':FN}, 
    {'True Negatives':TN_n, 'True Positives':TP_n, 'False Positives':FP_n, 'False Negatives':FN_n})

def extract_max_certainity(network_output):
    """
    Extracts the most probable outcome of a multi oracle network prediction.
    Params:
        network_output (pd.DataFrame): data containig the probabilities 
                                       of each outcome.
    Returns:
        the most probable outcome of  a multi oracle network prediction.
    """
    return pd.DataFrame(network_output).max(1)   


