import pandas as pd


def preparation(data_file):
    # read student data
    data = pd.read_csv(data_file)
    print('Data successfully loaded!')
    print(data.shape)
    print(data.head())

    # Data exploration
    students_num = data.shape[0]
    features_num = data.shape[1] - 1

    passed = data.loc[data.passed == 'yes', 'passed']
    passed_num = passed.shape[0]

    failed = data.loc[data.passed == 'no', 'passed']
    failed_num = failed.shape[0]

    print('Total students number:', students_num)
    print('Passed students number:', passed_num)
    print('Failed students number:', failed_num)
    print('Passed Percent: {:.2f}%'.format(passed_num * 100.0 / students_num))

    # Data Preparation
    feature_col = list(data.columns[: -1])
    label_col = data.columns[-1]

    features = data[feature_col]
    labels = data[label_col]

    return features, labels
