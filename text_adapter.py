"""
The text_adapter module contains functions
for preprocessing mock "emails" to create a Numpy
array suitable for usage with scikit-learn
Author: Camilla Montonen (2014)
Compatible with Python 2.X
"""

try:
    import numpy as np
except:
    print "Please make sure numpy is installed on your system"

def dictionary_builder(textdata):
    """
    Returns a list object of all unique words in text training data
    Argument: dict of strings (textdata) and their classes
    Returns: list of unique words
    """
    email_dictionary=[]
    for email in textdata:
        email_word=email.split()
        for word in email_word:
            if word.lower() not in email_dictionary:
                email_dictionary.append(word.lower())
            else:
                pass
    return email_dictionary



def binary_feature_vector_builder(dictionary, email):
    """
    Outputs a feature vector given a dictionary created from a text corpus
    and an email

    Arguments:
    -a list object containing all unique words in training data
    -a string ("the email")


    """
    feature_vector=[]
    email_words=email.split()
    for word in dictionary:
        if word in email_words:
            feature_vector.append(1)
        else:
            feature_vector.append(0)
    return feature_vector

def create_scikit_matrix(textdata):
    """
    Creates a scikit-learn numpy matrix where each sample is a row and each
    column a feature vector

    """
    #initialise the main matrix container
    matrix=[]
    #create a dictionary
    dictionary=dictionary_builder(textdata)
    #loop through each email, create a feature vector and append to matrix
    for email in textdata:
        temp_feature_vector=binary_feature_vector_builder(dictionary, email)
        matrix.append(temp_feature_vector)
    return np.array(matrix)
