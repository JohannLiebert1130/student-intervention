from time import time

from sklearn.metrics import f1_score

from data_preparation import preparation
from data_preprocess import preproces

features, labels = preparation('student-data.csv')
features_train, features_test, labels_train, labels_test = preproces(features, labels)

print('Training set has {} samples'.format(features_train.shape[0]))
print('Testing set has {} samples'.format(features_test.shape[0]))


def train_classifier(clf, X_train, y_train):
    # Fits a classifier to the training data.

    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()

    # Print the results
    print("Trained model in {:.4f} seconds".format(end - start))


def predict_labels(clf, features, target):
    # Makes predictions using a fit classifier based on F1 score.

    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    end = time()

    # Print and return results
    print("Made predictions in {:.4f} seconds.".format(end - start))
    return f1_score(target.values, y_pred, pos_label='yes')


def train_predict(clf, X_train, y_train, X_test, y_test):
    # Train and predict using a classifer based on F1 score.

    # Indicate the classifier and the training set size
    print("")
    print("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train)))

    # Train the classifier
    train_classifier(clf, X_train, y_train)

    # Print the results of prediction for both training and testing
    print("F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train)))
    print("F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test)))


# TODO: Import the three supervised learning models from sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# TODO: Initialize the three models
clf_A = GaussianNB()
clf_B = LogisticRegression(random_state=42)
clf_C = SVC(random_state=42)

# TODO: Execute the 'train_predict' function for each classifier and each training set size
# train_predict(clf, X_train, y_train, X_test, y_test)

for clf in [clf_A, clf_B, clf_C]:
    print("\n{}: ".format(clf.__class__.__name__))
    for n in [100, 200, 300]:
        train_predict(clf, features_train[:n], labels_train[:n], features_test, labels_test)

# Model Tuning (Logistic Regression)
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import make_scorer

''' We can use a stratified shuffle split data-split which preserves
    the percentage of samples for each class and combines it with cross validation. 
    This could be extremely useful when the dataset is 
    strongly imbalanced towards one of the two target labels'''
# Create the parameters list you wish to tune

C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
solver = ['sag']
max_iter = [1000]
param_grid = dict(C=C, solver=solver, max_iter=max_iter)

# Initialize the classifier
clf = LogisticRegression(random_state=42)

# Make an f1 scoring function using 'make_scorer'
f1_scorer = make_scorer(f1_score, pos_label='yes')

# Stratified Shuffle Split
ssscv = StratifiedShuffleSplit(test_size=0.1)

# TODO: Perform grid search on the classifier using the f1_scorer as the scoring method
grid_obj = GridSearchCV(clf, param_grid, cv=ssscv, scoring=f1_scorer)

# TODO: Fit the grid search object to the training data and find the optimal parameters
grid_obj = grid_obj.fit(features_train, labels_train)

# Get the estimator
clf = grid_obj.best_estimator_

# Report the final F1 score for training and testing after parameter tuning
print("Tuned model has a training F1 score of {:.4f}.".format(predict_labels(clf, features_train, labels_train)))
print("Tuned model has a testing F1 score of {:.4f}.".format(predict_labels(clf, features_test, labels_test)))
