import pandas as pd
from sklearn.model_selection import train_test_split


def preproces(features, labels):
    ''' Preprocesses the student data and converts non-numeric binary
    variables into binary variables. Converts categorical variables into
    dummy variabels. '''

    output = pd.DataFrame(index=features.index)

    for col_name, col_data in features.iteritems():
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix=col_name)

        output = output.join(col_data)

    ''' split data into training set and test set
    Use 300 training points (approximately 75%) and 95 testing points (approximately 25%).'''
    features_train, features_test, labels_train, labels_test = train_test_split(output, labels, stratify=labels,
                                                                                test_size=95, random_state=42)

    return features_train, features_test, labels_train, labels_test
