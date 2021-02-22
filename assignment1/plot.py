import matplotlib.pyplot as plt
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, \
    recall_score, average_precision_score, f1_score, plot_confusion_matrix, r2_score
from sklearn.model_selection import (validation_curve)
from sklearn.neural_network import MLPClassifier
from yellowbrick.model_selection import LearningCurve, ValidationCurve


def count_plot(data, label1, label2, title, dataset_name):
    plt.figure(figsize=(10, 6))
    labels = [label1, label2]
    plt.title(title, fontsize=14)
    ax = sns.countplot(data, palette='husl')
    ax.set_xticklabels(labels)
    plt.savefig("./plot_output/{}_{}_count_plot.png".format(dataset_name, title))
    plt.clf()



def bar_plot(data_x, data_y, title, dataset_name):
    plt.figure(figsize=(10, 6))
    plt.title(title, fontsize=14)
    sns.countplot(x=data_x, hue=data_y, palette='husl')
    plt.tight_layout()
    plt.savefig("./plot_output/{}_{}_bar_plot.png".format(dataset_name, title))
    plt.clf()


def corr_plot(data, dataset_name):
    corr = data.corr()
    corr.shape
    plt.subplots(figsize=(100, 100))
    ax = sns.heatmap(
        corr,
        vmin=-1, vmax=1, center=0,
        cmap='coolwarm',
        square=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=90,
        horizontalalignment='right'
    )
    plt.savefig("./plot_output/{}_correlation_plot.png".format(dataset_name))
    plt.clf()


def generate_validation_curve(model, clf_name, param_name, param_range, scoring, cv, dataset_name, X_train, y_train):
    if 'svm' in clf_name or 'nn' == clf_name:
        train_scores, test_scores = validation_curve(
            model, X_train, y_train, param_name=param_name, param_range=param_range,
            scoring="accuracy", n_jobs=8)
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        plt.figure(figsize=(10, 6))
        plt.title("Validation Curve with {}".format(clf_name))
        plt.xlabel(param_name)
        plt.ylabel("Score")
        plt.semilogx(param_range, train_scores_mean, label="Training score", marker='o', color="#0272a2")
        plt.semilogx(param_range, test_scores_mean, label="Cross-validation score", marker='o', color="#9fc377")
        plt.legend(loc="best")
        plt.savefig("./plot_output/{}_model_complexity_{}_{}.png".format(clf_name, dataset_name, param_name))
        plt.clf()

    else:
        plt.figure(figsize=(10, 6))
        viz = ValidationCurve(model, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)
        viz.fit(X_train, y_train)
        viz.show("./plot_output/{}_model_complexity_{}_{}.png".format(clf_name, dataset_name, param_name))
        plt.savefig("./plot_output/{}_model_complexity_{}_{}.png".format(clf_name, dataset_name, param_name))
        plt.clf()


def generate_learning_curve(model, clf_name, scoring, sizes, cv, n_jobs, dataset_name, X_train, y_train):
    plt.figure(figsize=(10, 6))
    viz = LearningCurve(model, cv=cv, scoring=scoring, train_sizes=sizes, n_jobs=n_jobs)
    viz.fit(X_train, y_train)
    viz.show("./plot_output/{}_learning_curve_{}_{}.png".format(clf_name, dataset_name,scoring))
    plt.savefig("./plot_output/{}_learning_curve_{}_{}.png".format(clf_name, dataset_name,scoring))
    plt.clf()


def record_metrics(clf_name, ds_name, best_params, before_tuned_score, after_tuned_score, scoring, train_time,
                   query_time,window_score,query_time_window):
    filename = "./plot_output/metrics.csv".format(clf_name)
    with open(filename, 'a+') as f:
        timestamp = time.time()
        f.write("{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}\n".format(timestamp, clf_name, ds_name, best_params, before_tuned_score,
                                                      after_tuned_score, scoring, train_time, query_time, window_score, query_time_window))


def generate_nn_curves(clf_name, clf, dataset, loss_curve, X_train, y_train, X_test, y_test):
    # Following code was taken from the users TomDLT and Chenn on Stack Overflow
    # https://stackoverflow.com/questions/46912557/is-it-possible-to-get-test-scores-for-each-iteration-of-mlpclassifier
    plt.figure(figsize=(10, 6))
    plt.title("Loss Curve for {}".format(dataset))
    plt.xlabel("epoch")
    plt.plot(loss_curve)
    plt.savefig("./plot_output/{}_loss_curve_{}.png".format(clf_name, dataset))
    plt.clf()

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    """ Home-made mini-batch learning
    -> not to be used in out-of-core setting!
    """
    N_TRAIN_SAMPLES = X_train.shape[0]
    N_EPOCHS = 1200
    N_BATCH = 50
    N_CLASSES = np.unique(y_train)

    scores_train = []
    scores_test = []

    clf = MLPClassifier(**clf.get_params())

    # EPOCH
    epoch = 0
    while epoch < N_EPOCHS:
        # SHUFFLING
        random_perm = np.random.permutation(X_train.shape[0])
        mini_batch_index = 0
        while True:
            # MINI-BATCH
            indices = random_perm[mini_batch_index:mini_batch_index + N_BATCH]
            clf.partial_fit(X_train[indices], y_train[indices], classes=N_CLASSES)
            mini_batch_index += N_BATCH

            if mini_batch_index >= N_TRAIN_SAMPLES:
                break

        # SCORE TRAIN
        scores_train.append(clf.score(X_train, y_train))

        # SCORE TEST
        scores_test.append(clf.score(X_test, y_test))

        epoch += 1

    """ Plot """
    plt.figure(figsize=(10, 6))
    plt.plot(scores_train, alpha=0.8, label="Training score", color="#013d56")
    plt.plot(scores_test, alpha=0.8, label="Cross-validation score", color="#a54962")
    plt.title("Accuracy over epochs", fontsize=14)
    plt.xlabel('Epochs')
    plt.legend(loc='best')
    plt.savefig("./plot_output/{}_accuracy_over_epochs_{}.png".format(clf_name, dataset))
    plt.clf()


def evaluate(y_pred, y_test, initial_score, clf_name, dataset_name):
    initial_accuracy = initial_score
    accuracy = accuracy_score(y_test, y_pred)
    error_score = r2_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    pr_auc = average_precision_score(y_test, y_pred)
    f_score = f1_score(y_test, y_pred)

    result = pd.DataFrame(
        [[dataset_name, clf_name, initial_accuracy, accuracy, error_score, precision, recall, roc_auc, pr_auc, f_score]],
        columns=['DataSetName', 'Model', 'Initial Accuracy Train', 'Accuracy','R2','Precision', 'Recall', 'ROC_auc',
                 'PR_auc', 'F1-Score'])
    return (result)


def confusion_matrix_plot(clf, X_test, y_test, clf_name, dataset_name):
    plot_confusion_matrix(clf, X_test, y_test)
    plt.savefig("./plot_output/{}_confusion_matrix_{}.png".format(clf_name, dataset_name))
    plt.clf()


def plot_roc(result_table, dataset_name):
    fig = plt.figure(figsize=(8, 6))

    for i in result_table.index:
        plt.plot(result_table.loc[i]['fpr'],
                 result_table.loc[i]['tpr'],
                 label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))

    plt.plot([0, 1], [0, 1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("Flase Positive Rate", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)

    plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
    plt.legend(prop={'size': 13}, loc='lower right')
    plt.savefig("./plot_output/{}_roc_curve.png".format(dataset_name))
    plt.clf()
    plt.show()
