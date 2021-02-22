import time
import warnings
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.model_selection import (GridSearchCV, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from data_preprocessing import covid_data_load_data, credit_card_load_data, covid_data_window2_load_data
from plot import generate_validation_curve, generate_learning_curve, generate_nn_curves, record_metrics, evaluate, \
    plot_roc, confusion_matrix_plot

warnings.filterwarnings("ignore")
random_st = 123
experiments = ['ada', 'dt', 'knn', 'nn', 'svm']
results = pd.DataFrame(
    columns=['DataSetName', 'Model', 'Initial Accuracy Train', 'Accuracy', 'R2', 'Precision', 'Recall', 'ROC_auc',
             'PR_auc', 'F1-Score'])
result_table = pd.DataFrame(columns=['classifiers', 'fpr', 'tpr', 'auc'])
verb = True
max_depths = np.arange(1, 11, 1)
n_neighbors = np.arange(1, 206, 2)

samples = 100
features = 10

gamma_fracs = np.arange(1 / features, 2.1, 0.2)
tols = np.arange(1e-8, 1e-1, 0.01)
C_values = np.arange(0.001, 2.5, 0.25)
iters = [-1, int((1e6 / samples) / .8) + 1]

alphas = [10 ** -x for x in np.arange(-1, 9.01, 0.5)]
d = 10
hiddens = [(h,) * l for l in [1, 2, 3] for h in [d, d // 2, d * 2]]
learning_rates = sorted([(2 ** x) / 1000 for x in range(8)] + [0.000001])

# Define parameters for tuning:
dt_params_dict = {"criterion": ["gini", "entropy"],
                  "class_weight": ["balanced", None],
                  'max_depth': np.arange(10, 200, 10),
                  'min_samples_leaf': range(1, 25)}

ada_params_dict = {'n_estimators': [1, 2, 5, 10, 20, 30, 45, 60, 80, 90, 100],
                   'learning_rate': [(2 ** x) / 100 for x in range(7)] + [1],
                   'base_estimator__max_depth': max_depths}

knn_params_dict = {'metric': ['manhattan', 'euclidean', 'chebyshev'], 'n_neighbors': n_neighbors,
                   'weights': ['uniform', 'distance']}

svm_rbf_params_dict = {'max_iter': iters, 'tol': tols, 'class_weight': ['balanced', None],
                       'kernel': ['rbf'],
                       'C': C_values}

svm_sigmoid_params_dict = {'max_iter': iters, 'tol': tols, 'class_weight': ['balanced', None],
                           'kernel': ['sigmoid'],
                           'C': C_values}

nn_params_dict = {'activation': ['relu', 'logistic'],
                  'alpha': alphas,
                  'learning_rate_init': learning_rates,
                  'hidden_layer_sizes': hiddens}

# Classification models dictionary
CLASSIFIERS_DICT = {'dt': {'clf': DecisionTreeClassifier,
                           'tuning_params': dt_params_dict, 'params': {}},

                    'ada': {'clf': AdaBoostClassifier,
                            'tuning_params': ada_params_dict,
                            'params': {
                                'base_estimator': DecisionTreeClassifier(criterion='entropy', class_weight='balanced',
                                                                         max_depth=10,
                                                                         random_state=random_st)}},
                    'knn': {'clf': KNeighborsClassifier,
                            'tuning_params': knn_params_dict,
                            'params': {}},
                    'svm_rbf': {'clf': SVC,
                                'tuning_params': svm_rbf_params_dict,
                                'params': {'kernel': 'rbf', 'verbose': False, 'probability': True}},
                    'svm_sigmoid': {'clf': SVC,
                                    'tuning_params': svm_sigmoid_params_dict,
                                    'params': {'kernel': 'sigmoid', 'verbose': False, 'probability': True}},
                    'nn': {'clf': MLPClassifier,
                           'tuning_params': nn_params_dict,
                           'params': {'max_iter': 200, 'hidden_layer_sizes': (5, 2), 'activation': 'logistic',
                                      'verbose': False}}
                    }
# Best parameters for "Covid-19 dataset"
CLASSIFIERS_DICT_Best_DS1 = {'nn': {'clf': MLPClassifier,
                                    'params': {'max_iter': 200, 'activation': 'logistic', 'alpha': 1e-06,
                                               'hidden_layer_sizes': (20,), 'learning_rate_init': 0.002}},
                             'knn': {'clf': KNeighborsClassifier,
                                     'params': {'metric': 'manhattan', 'n_neighbors': 11, 'weights': 'distance'}},
                             'svm_rbf': {'clf': SVC,
                                         'tuning_params': svm_rbf_params_dict,
                                         'params': {'C': 2.251, 'class_weight': 'balanced', 'kernel': 'rbf',
                                                    'max_iter': -1, 'tol': 0.01000001, 'verbose': False,
                                                    'probability': True}},
                             'svm_sigmoid': {'clf': SVC,
                                             'params': {'C': 0.251, 'class_weight': None, 'kernel': 'sigmoid',
                                                        'max_iter': -1, 'tol': 0.050000010000000004, 'verbose': False,
                                                        'probability': True}},
                             'dt': {'clf': DecisionTreeClassifier,
                                    'params': {'class_weight': None, 'criterion': 'entropy', 'max_depth': 10,
                                               'min_samples_leaf': 1}},

                             'ada': {'clf': AdaBoostClassifier,
                                     'params': {'learning_rate': 0.04, 'n_estimators': 80,
                                                'base_estimator': DecisionTreeClassifier(criterion='entropy',
                                                                                         class_weight='balanced',
                                                                                         max_depth=4,
                                                                                         random_state=random_st)}}
                             }
# Best parameters for "Credit Card"
CLASSIFIERS_DICT_Best_DS2 = {'nn': {'clf': MLPClassifier,
                                    'params': {'max_iter': 200, 'activation': 'relu', 'alpha': 0.31622776601683794,
                                               'hidden_layer_sizes': (5,), 'learning_rate_init': 0.008}},
                             'knn': {'clf': KNeighborsClassifier,
                                     'params': {'metric': 'manhattan', 'n_neighbors': 3, 'weights': 'uniform'}},
                             'svm_rbf': {'clf': SVC,
                                         'params': {'C': 0.501, 'class_weight': None, 'kernel': 'rbf', 'max_iter': -1,
                                                    'tol': 0.07000001, 'verbose': False, 'probability': True}},
                             'svm_sigmoid': {'clf': SVC,
                                             'params': {'C': 0.501, 'class_weight': None, 'kernel': 'sigmoid',
                                                        'max_iter': -1, 'tol': 0.06000001, 'verbose': False,
                                                        'probability': True}},
                             'dt': {'clf': DecisionTreeClassifier,
                                    'params': {'class_weight': 'balanced', 'criterion': 'gini', 'max_depth': 10,
                                               'min_samples_leaf': 23}},

                             'ada': {'clf': AdaBoostClassifier,
                                     'params': {'learning_rate': 1, 'n_estimators': 100,
                                                'base_estimator': DecisionTreeClassifier(criterion='entropy',
                                                                                         class_weight='balanced',
                                                                                         max_depth=5,
                                                                                         random_state=random_st)}}
                             }


def run_experiments(experiment=None, all=False):
    global samples
    global features
    global d
    global result_table
    global results
    datasets = ['Credit_Card_Approval', 'Covid-19_ICU_Admission']

    for dataset in datasets:
        if dataset == 'Covid-19_ICU_Admission':
            print("Load and preprocess Covid-19_ICU_Admission dataset")
            df = covid_data_load_data()
        if dataset == 'Credit_Card_Approval':
            print("Load and preprocess Credit_Card_Approval dataset")
            df = credit_card_load_data()

        x, y = df.iloc[:, :-1], df.iloc[:, -1]
        samples = x.shape[0]
        features = x.shape[1]
        d = x.shape[1]
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=random_st)
        # Run all experiments
        if all:
            result_table = result_table[0:0]
            print("---------------------")
            print("Running Models ......")
            for clf_name, obj in CLASSIFIERS_DICT.items():
                run_model(X_train, X_test, y_train, y_test, dataset, clf_name)
                print_results(results)

            result_table.set_index('classifiers', inplace=True)
            plot_roc(result_table, dataset)
        # Run single experiments
        elif experiment in experiments:
            result_table = result_table[0:0]
            if experiment == 'svm':
                run_model(X_train, X_test, y_train, y_test, dataset, 'svm_sigmoid')
                print_results(results)
                run_model(X_train, X_test, y_train, y_test, dataset, 'svm_rbf')
                print_results(results)
            else:
                run_model(X_train, X_test, y_train, y_test, dataset, experiment)
                print_results(results)
            result_table.set_index('classifiers', inplace=True)
            plot_roc(result_table, dataset)

        results.to_csv("./plot_output/{}_results.csv".format(dataset), index=False, sep=',', encoding='utf-8')
        results = results[0:0]


def run_best_models(X_train, X_test, y_train, y_test, dataset_name, clf_name):
    global results
    if 'svm' in clf_name:
        X_train = preprocessing.scale(X_train)
        X_test = preprocessing.scale(X_test)
    if dataset_name == 'Covid-19_ICU_Admission':
        clf_dict = CLASSIFIERS_DICT_Best_DS1[clf_name]
    else:
        clf_dict = CLASSIFIERS_DICT_Best_DS2[clf_name]

    clf_obj = clf_dict['clf']
    params = clf_dict['params']
    print('-------------------------------------------------')
    print("Running Best {} Learner for {}".format(clf_name, dataset_name))
    clf = clf_obj(**params)
    # Train model
    start_time = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start_time
    print("Train time: {}".format(train_time))
    print()
    # Predict
    start_time = time.time()
    y_pred = clf.predict(X_test)
    query_time = time.time() - start_time

    print("Query time: {}".format(query_time))
    print()
    # Print Results
    current_result = evaluate(y_pred, y_test, 0, clf_name, dataset_name)
    results = results.append(current_result)


def run_experiments_best():
    global results
    datasets = ['Covid-19_ICU_Admission', 'Credit_Card_Approval']
    for dataset in datasets:
        if dataset == 'Covid-19_ICU_Admission':
            print("Load and preprocess Covid-19_ICU_Admission dataset")
            df = covid_data_load_data()
        if dataset == 'Credit_Card_Approval':
            print("Load and preprocess Credit_Card_Approval dataset")
            df = credit_card_load_data()
        x, y = df.iloc[:, :-1], df.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=random_st)
        print("---------------------")
        print("Running Models Expretiments  ......")
        for clf_name, obj in CLASSIFIERS_DICT.items():
            run_best_models(X_train, X_test, y_train, y_test, dataset, clf_name)
            print_results(results)
    return 0


def run_model(X_train, X_test, y_train, y_test, dataset_name, clf_name):
    # Define a result table as a DataFrame
    global result_table
    window_score = 0
    # FOR SVM NEED TO SCALE DATA
    if 'svm' in clf_name:
        X_train = preprocessing.scale(X_train)
        X_test = preprocessing.scale(X_test)
    clf_dict = CLASSIFIERS_DICT[clf_name]
    clf_obj = clf_dict['clf']
    tuning_params = clf_dict['tuning_params']
    params = clf_dict['params']
    # dataset_name = dataset.split('.')[0]

    print("Running {} Learner for {}".format(clf_name, dataset_name))

    # Initial accuracy without tuning parameters
    clf = clf_obj(**params)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cv = 5

    initial_score = accuracy_score(y_test, y_pred)
    print("Initial score {}: {} ".format(clf_name, initial_score))

    # Validation Curve with default classifier
    scoring = 'accuracy'
    for key, value in tuning_params.items():
        if key == 'hidden_layer_sizes' or key == 'metric' or key == 'class_weight':
            continue
        generate_validation_curve(clf_obj(**params), clf_name, key, value, scoring, cv, dataset_name, X_train, y_train)

    # Grid Search to obtain best parameters
    clf = clf_obj(**params)
    grid_params = [tuning_params]
    tuned_clf = GridSearchCV(clf, grid_params, scoring=scoring, cv=cv, verbose=verb, n_jobs=-1)
    tuned_clf.fit(X_train, y_train)

    print("Best parameters set found on development set: {}".format(tuned_clf.best_params_))
    print()

    # Generate Learning Curve with scorint accuracy and r_2
    clf = clf_obj(**params)
    clf.set_params(**tuned_clf.best_params_)
    sizes = np.linspace(0.1, 1.0, 10)
    generate_learning_curve(clf, clf_name, scoring, sizes, cv, 8, dataset_name, X_train, y_train)
    generate_learning_curve(clf, clf_name, 'r2', sizes, cv, 8, dataset_name, X_train, y_train)

    # Fit best parameters
    final_clf = clf_obj()
    final_clf.set_params(**params)
    final_clf.set_params(**tuned_clf.best_params_)
    start_time = time.time()
    final_clf.fit(X_train, y_train)
    train_time = time.time() - start_time

    print("Train time: {}".format(train_time))
    print()

    # Predict with  best parameters
    start_time = time.time()
    y_pred = final_clf.predict(X_test)
    query_time = time.time() - start_time

    print("Query time: {}".format(query_time))
    print()

    # Accuracy after tuning hyperparameters
    final_score = accuracy_score(y_test, y_pred)

    # Record the results for confusion matrix
    record_results_confusion_matrix(X_test, clf_name, final_clf, y_test)

    # Record model results
    current_result = evaluate(y_pred, y_test, initial_score, clf_name, dataset_name)
    global results
    results = results.append(current_result)
    confusion_matrix_plot(final_clf, X_test, y_test, clf_name, dataset_name)

    # Test for window "0-2"
    query_time_window = 0
    if dataset_name == 'Covid-19_ICU_Admission':
        df = covid_data_window2_load_data()
        X_test, y_test = df.iloc[:, :-1], df.iloc[:, -1]
        if 'svm' in clf_name:
            X_test = preprocessing.scale(X_test)
        y_pred = final_clf.predict(X_test)
        window_score = accuracy_score(y_test, y_pred)
        record_results_confusion_matrix(X_test, 'Covid-19_ICU_Admission_Window_0_2', final_clf, y_test)
        confusion_matrix_plot(final_clf, X_test, y_test, 'Covid-19_ICU_Admission_Window_0_2', dataset_name)

        print("Window 0-2 Tuned Score: {}".format(window_score))

    record_metrics(clf_name, dataset_name, tuned_clf.best_params_, initial_score, final_score, scoring, train_time,
                   query_time, window_score, query_time_window)
    # Generate loss curves for neural network
    if clf_name == 'nn':
        generate_nn_curves(clf_name, final_clf, dataset_name, final_clf.loss_curve_, X_train, y_train, X_test, y_test)


def record_results_confusion_matrix(X_test, clf_name, final_clf, y_test):
    global result_table
    yproba = final_clf.predict_proba(X_test)[::, 1]
    fpr, tpr, _ = roc_curve(y_test, yproba)
    auc = roc_auc_score(y_test, yproba)
    result_table = result_table.append({'classifiers': clf_name,
                                        'fpr': fpr,
                                        'tpr': tpr,
                                        'auc': auc}, ignore_index=True)


def print_results(df):
    print('|||||||||||||||||||||||||||||||||||||||||||||||||||||||')
    print(
        '[ Experiment Results for dataset {}, Model {}]'.format(df.iloc[-1, :]["DataSetName"], df.iloc[-1, :]["Model"]))
    print('Accuracy:   {}'.format(df.iloc[-1, :]["Accuracy"]))
    print('R2 ERROR:  {}'.format(df.iloc[-1, :]["R2"]))
    print('Precision:  {}'.format(df.iloc[-1, :]["Precision"]))
    print('Recall:     {}'.format(df.iloc[-1, :]["Recall"]))
    print('ROC Auc:    {}'.format(df.iloc[-1, :]["ROC_auc"]))
    print('PR Auc:     {}'.format(df.iloc[-1, :]["PR_auc"]))
    print('F1-Score:   {}'.format(df.iloc[-1, :]["F1-Score"]))
    print('|||||||||||||||||||||||||||||||||||||||||||||||||||||||')
