import copy
import logging
import pandas as pd
import numpy as np

from collections import Counter

from sklearn import preprocessing, utils
import sklearn.model_selection as ms
from scipy.sparse import isspmatrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import seaborn as sns
import plotting as pl

from abc import ABC, abstractmethod

# TODO: Move this to a common lib?
OUTPUT_DIRECTORY = './output_OLD'

if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)
if not os.path.exists('{}/images'.format(OUTPUT_DIRECTORY)):
    os.makedirs('{}/images'.format(OUTPUT_DIRECTORY))
if not os.path.exists('{}/plots'.format(OUTPUT_DIRECTORY)):
    os.makedirs('{}/plots'.format(OUTPUT_DIRECTORY))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def plot_pairplot(title, df, class_column_name=None):
    plt = sns.pairplot(df, hue=class_column_name)
    return plt


# Adapted from https://stats.stackexchange.com/questions/239973/a-general-measure-of-data-set-imbalance
def is_balanced(seq):
    n = len(seq)
    classes = [(clas, float(count)) for clas, count in Counter(seq).items()]
    k = len(classes)

    H = -sum([(count / n) * np.log((count / n)) for clas, count in classes])
    return H / np.log(k) > 0.75


class DataLoader(ABC):
    def __init__(self, path, verbose, seed):
        self._path = path
        self._verbose = verbose
        self._seed = seed

        self.features = None
        self.classes = None
        self.testing_x = None
        self.testing_y = None
        self.training_x = None
        self.training_y = None
        self.binary = False
        self.balanced = False
        self._data = pd.DataFrame()

    def load_and_process(self, data=None, preprocess=True):
        """
        Load data from the given path and perform any initial processing required. This will populate the
        features and classes and should be called before any processing is done.

        :return: Nothing
        """
        if data is not None:
            self._data = data
            self.features = None
            self.classes = None
            self.testing_x = None
            self.testing_y = None
            self.training_x = None
            self.training_y = None
        else:
            self._load_data()
        self.log("Processing {} Path: {}, Dimensions: {}", self.data_name(), self._path, self._data.shape)
        if self._verbose:
            old_max_rows = pd.options.display.max_rows
            pd.options.display.max_rows = 10
            self.log("Data Sample:\n{}", self._data)
            pd.options.display.max_rows = old_max_rows

        if preprocess:
            self.log("Will pre-process data")
            self._preprocess_data()

        self.get_features()
        self.get_classes()
        self.log("Feature dimensions: {}", self.features.shape)
        self.log("Classes dimensions: {}", self.classes.shape)
        self.log("Class values: {}", np.unique(self.classes))
        class_dist = np.histogram(self.classes)[0]
        class_dist = class_dist[np.nonzero(class_dist)]
        self.log("Class distribution: {}", class_dist)
        self.log("Class distribution (%): {}", (class_dist / self.classes.shape[0]) * 100)
        self.log("Sparse? {}", isspmatrix(self.features))

        if len(class_dist) == 2:
            self.binary = True
        self.balanced = is_balanced(self.classes)

        self.log("Binary? {}", self.binary)
        self.log("Balanced? {}", self.balanced)

    def scale_standard(self):
        self.features = StandardScaler().fit_transform(self.features)
        if self.training_x is not None:
            self.training_x = StandardScaler().fit_transform(self.training_x)

        if self.testing_x is not None:
            self.testing_x = StandardScaler().fit_transform(self.testing_x)

    def build_train_test_split(self, test_size=0.3):
        if not self.training_x and not self.training_y and not self.testing_x and not self.testing_y:
            self.training_x, self.testing_x, self.training_y, self.testing_y = ms.train_test_split(
                self.features, self.classes, test_size=test_size, random_state=self._seed, stratify=self.classes
            )

    def get_features(self, force=False):
        if self.features is None or force:
            self.log("Pulling features")
            self.features = np.array(self._data.iloc[:, 0:-1])

        return self.features

    def get_classes(self, force=False):
        if self.classes is None or force:
            self.log("Pulling classes")
            self.classes = np.array(self._data.iloc[:, -1])

        return self.classes

    def dump_test_train_val(self, test_size=0.2, random_state=123):
        ds_train_x, ds_test_x, ds_train_y, ds_test_y = ms.train_test_split(self.features, self.classes,
                                                                           test_size=test_size,
                                                                           random_state=random_state,
                                                                           stratify=self.classes)
        pipe = Pipeline([('Scale', preprocessing.StandardScaler())])
        train_x = pipe.fit_transform(ds_train_x, ds_train_y)
        train_y = np.atleast_2d(ds_train_y).T
        test_x = pipe.transform(ds_test_x)
        test_y = np.atleast_2d(ds_test_y).T

        train_x, validate_x, train_y, validate_y = ms.train_test_split(train_x, train_y,
                                                                       test_size=test_size, random_state=random_state,
                                                                       stratify=train_y)
        test_y = pd.DataFrame(np.where(test_y == 0, -1, 1))
        train_y = pd.DataFrame(np.where(train_y == 0, -1, 1))
        validate_y = pd.DataFrame(np.where(validate_y == 0, -1, 1))

        tst = pd.concat([pd.DataFrame(test_x), test_y], axis=1)
        trg = pd.concat([pd.DataFrame(train_x), train_y], axis=1)
        val = pd.concat([pd.DataFrame(validate_x), validate_y], axis=1)

        tst.to_csv('data/{}_test.csv'.format(self.data_name()), index=False, header=False)
        trg.to_csv('data/{}_train.csv'.format(self.data_name()), index=False, header=False)
        val.to_csv('data/{}_validate.csv'.format(self.data_name()), index=False, header=False)

    @abstractmethod
    def _load_data(self):
        pass

    @abstractmethod
    def data_name(self):
        pass

    @abstractmethod
    def _preprocess_data(self):
        pass

    @abstractmethod
    def class_column_name(self):
        pass

    @abstractmethod
    def pre_training_adjustment(self, train_features, train_classes):
        """
        Perform any adjustments to training data before training begins.
        :param train_features: The training features to adjust
        :param train_classes: The training classes to adjust
        :return: The processed data
        """
        return train_features, train_classes

    def reload_from_hdf(self, hdf_path, hdf_ds_name, preprocess=True):
        self.log("Reloading from HDF {}".format(hdf_path))
        loader = copy.deepcopy(self)

        df = pd.read_hdf(hdf_path, hdf_ds_name)
        loader.load_and_process(data=df, preprocess=preprocess)
        loader.build_train_test_split()

        return loader

    def log(self, msg, *args):
        """
        If the learner has verbose set to true, log the message with the given parameters using string.format
        :param msg: The log message
        :param args: The arguments
        :return: None
        """
        if self._verbose:
            logger.info(msg.format(*args))


class CreditCardApproval(DataLoader):

    def __init__(self, path='data/CreditCardApproval.data', verbose=False, seed=1):
        # Uncomment to run the loader
        # def __init__(self, path='CreditCardApproval.data', verbose=False, seed=1):
        super().__init__(path, verbose, seed)

    def _load_data(self):
        # self._data = pd.read_excel(self._path, header=1, index_col=0)

        self._data = pd.read_csv(self._path, header=None)

    def data_name(self):
        return 'Credit_Card_Approval'

    def class_column_name(self):
        return 'Credit_Card_Approval'

    def check_for_correlation(self):
        corr = self._data.corr()
        corr_data = corr.unstack().sort_values().drop_duplicates()
        ind = corr_data[corr_data.between(0.81, 0.999999)].index
        corr_ind = []
        for i in ind:
            if i == 'WINDOW':
                continue
            corr_ind.append(i[0])
        return corr_ind

    def _preprocess_data(self):
        print("Analise Credit Card Approval  dataset:")
        print("Data has", len(self._data), "rows and", len(self._data.columns), "columns.")
        if self._data.isnull().values.any():
            print("Warning: Missing Data")

        # Replace the '?'s with NaN
        self._data = self._data.replace('?', np.nan)

        missing_ind = self._data.columns[self._data.isnull().any()]
        # Replace missing values with most frequent
        for ind in missing_ind:
            if self._data[ind].dtypes == 'object':
                self._data[ind] = self._data[ind].fillna(self._data[ind].value_counts().index[0])
        # convert second column to int and create bins
        self._data[1] = self._data[1].astype(float).astype(int)
        self._data[1] = pd.cut(x=self._data[1], bins=range(10, 100, 5), labels=range(15, 100, 5))
        # get numerical and categorical features.
        categorical = self._data.select_dtypes(include=['object']).columns
        numerical = self._data.select_dtypes(exclude=['object']).columns

        # Replace categorigal with numerical variables
        label_encoder = preprocessing.LabelEncoder()
        self._data[categorical] = self._data[categorical].apply(label_encoder.fit_transform)

        # Replace outliers
        self._data[14] = np.where(self._data[14] > 20000, 2000, self._data[14])
        self._data[10] = np.where(self._data[10] > 15, 12, self._data[10])
        self._data[7] = np.where(self._data[7] > 15, 12, self._data[7])

        print("Check for correlation.....")
        corr_ind = self.check_for_correlation()
        self._data = self._data.drop(corr_ind, axis=1)
        cols_to_norm = [2, 7, 13, 14]
        self._data[cols_to_norm] = self._data[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        print('Final data Rows: {} | Final data Columns: {}'.format(self._data.shape[0], self._data.shape[1]))

        self.features, self.classes = self._data.iloc[:, :-1], self._data.iloc[:, -1]
    def pre_training_adjustment(self, train_features, train_classes):
        """
        Perform any adjustments to training data before training begins.
        :param train_features: The training features to adjust
        :param train_classes: The training classes to adjust
        :return: The processed data
        """

        # Standardize
        # Create the Scaler object
        # scaler = preprocessing.StandardScaler()
        # # Fit data on the scaler object
        # train_features = scaler.fit_transform(train_features)

        # PCA
        # pca = PCA(n_components=0.80)
        # train_features = pca.fit_transform(train_features)
        # print(train_features.shape)

        return train_features, train_classes

    def analise_data(self):
        df = pd.read_csv(self._path)
        df.info()

        print("-------------------------------")
        print("Data has", len(df), "rows and", len(df.columns), "columns.")
        print("-------------------------------")

        print(df.describe())

        if df.isnull().values.any():
            print("Warning: Missing Data")
            print(self._data.isnull().sum())
        # Count target value distribution
        print("-------------------------------")
        print("---------Target value distribution----------")
        print("-------------------------------")
        count = df.iloc[:, -1].value_counts()
        print(count)

        print("---------Target value distribution in percents----------")

        percent = count / len(df) * 100
        print(percent)
        print("-------------------------------")


class Covid_19_ICU_Admission(DataLoader):

    def __init__(self, path='data/Covid_19_ICU_Admission.xlsx', verbose=False, seed=1):
        # Uncomment to run the loader
        # def __init__(self, path='Covid_19_ICU_Admission.xlsx', verbose=False, seed=1):
        super().__init__(path, verbose, seed)

    def _load_data(self):
        self._data = pd.read_excel(self._path)

    def data_name(self):
        return 'Covid_19_ICU_Admission'

    def class_column_name(self):
        return 'Covid_19_ICU_Admission'

    def check_for_correlation(self):
        corr = self._data.corr()
        corr_data = corr.unstack().sort_values().drop_duplicates()
        ind = corr_data[corr_data.between(0.81, 0.999999)].index
        corr_ind = []
        for i in ind:
            if i == 'WINDOW':
                continue
            corr_ind.append(i[0])
        return corr_ind

    def _preprocess_data(self):

        print("Data has", len(self._data), "rows and", len(self._data.columns), "columns.")
        if self._data.isnull().values.any():
            print("Warning: Missing Data")

        self._data['AGE_PERCENTIL'].replace(['10th', '20th', '30th', '40th', '50th', '60th', '70th', '80th',
                                       '90th', 'Above 90th'], [10, 20, 30, 40, 50, 60, 70, 80, 90, 91], inplace=True)
        self._data = self._data.dropna(axis=0, subset=['DISEASE GROUPING 1'])
        # Fill missing values by group using bfill and ffill
        self._data = self._data.groupby(['PATIENT_VISIT_IDENTIFIER'], as_index=False).apply(lambda group: group.bfill())
        self._data = self._data.groupby(['PATIENT_VISIT_IDENTIFIER'], as_index=False).apply(lambda group: group.ffill())

        # remove data wich has Nan values
        self._data=self._data.dropna()
        self._data=self._data.drop(['PATIENT_VISIT_IDENTIFIER', 'WINDOW'], axis=1)
        corr_ind = self.check_for_correlation()
        self._data = self._data.drop(corr_ind, axis=1)
        print('Final data Rows: {} | Final data Columns: {}'.format(self._data.shape[0], self._data.shape[1]))

        #self.features, self.classes = self._data.iloc[:, :-1], self._data.iloc[:, -1]

    def pre_training_adjustment(self, train_features, train_classes):
        """
        Perform any adjustments to training data before training begins.
        :param train_features: The training features to adjust
        :param train_classes: The training classes to adjust
        :return: The processed data
        """

        return train_features, train_classes

    def analise_data(self):
        df = pd.read_excel(self._path)
        df.info()

        print("-------------------------------")
        print("Data has", len(df), "rows and", len(df.columns), "columns.")
        print("-------------------------------")

        print(df.describe())

        if df.isnull().values.any():
            print("Warning: Missing Data")
            print(self._data.isnull().sum())
        # Count target value distribution
        print("-------------------------------")
        print("---------Target value distribution----------")
        print("-------------------------------")
        count = df.groupby('ICU').size()
        print(count)

        print("---------Target value distribution in percents----------")

        percent = count / len(df) * 100
        print(percent)
        print("-------------------------------")
        numeric_cols = df.select_dtypes(exclude=['object']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns


if __name__ == '__main__':
    cd_data = CreditCardApproval(verbose=True)
    cd_data.analise_data()
    cd_data.load_and_process()

    covid_data = Covid_19_ICU_Admission(verbose=True)
    covid_data.analise_data()
    covid_data.load_and_process()
