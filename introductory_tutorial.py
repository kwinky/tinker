"""
A Toy Spam Filter using the BernoulliNB classifier in Scikit-learn
Template
"""

#1.1 Import the BernoulliNB class from the Scikit-learn package
import numpy as np
#1.2. Import Numpy
import text_adapter
#1.3 Import the text_adapter
from sklearn.naive_bayes import BernoulliNB


#3. Here is the training data for your classifier
spam_emails=["Hello send your password", "hello please click link", "click link",
"your password here", "send password"]
ham_emails=["hello reset your password", "password email", "warm hello" ]

#4. Prepare your data

#concatenate the training data and assign it to a variable
training_data = spam_emails + ham_emails
#make a numpy array of training labels
training_labels = np.array([1]*len(spam_emails)+[0]*len(ham_emails))

#5. Generate the training data matrix, store it in a variable and print it out
training_data_matrix=text_adapter.create_scikit_matrix(training_data)
print training_data_matrix

#6. Create a classifier by calling an instance of BernoulliNB

classifier=BernoulliNB()

#7.Train the classifier by calling the fit function
classifier.fit(training_data_matrix, training_labels)

#Now you have a trained classifier!

#8. Test the classifier by calculating the probability that a new email is spam

import sys

print 'Number of arguments:', len(sys.argv), 'arguments.'
if len(sys.argv) > 1:
	test_email = str(sys.argv[1])
else:
	test_email="hello please send password"

print test_email

training_dictionary=text_adapter.dictionary_builder(training_data)
test_feature_vector=np.array(text_adapter.binary_feature_vector_builder(training_dictionary,test_email))

print "[not spam | spam]"
print classifier.predict_proba(test_feature_vector)


