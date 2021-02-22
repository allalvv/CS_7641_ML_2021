from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
import plot as pl


def covid_data_load_data():
    df = pd.read_excel("./data/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx")
    covid_data_analysis(df)
    final_data = covid_data_preprocessing(df)
    return final_data

def credit_card_load_data():
    df = pd.read_csv('./data/crx.data', header=None)
    credit_card_analysis(df)
    final_data = credit_card_preprocessing(df)
    return final_data


def covid_data_window2_load_data():
    df = pd.read_excel("./data/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx")
    final_data = covid_data_preprocessing(df, window=True)
    return final_data


def covid_data_analysis(data):
    print("Analise Covid-19 ICU Admission dataset:")
    # Print data shape
    print('Rows: {} | Columns: {}'.format(data.shape[0], data.shape[1]))
    # Check imbalance
    print("Class distribution:  ")
    print("-------------------------------")
    print(data['ICU'].value_counts())
    print("-------------------------------")

    print("Plotting data ...")
    pl.count_plot(data['ICU'], "Not Admited to ICU", "Admited to ICU", 'ICU Admission', "Covid-19_ICU_Admission_")
    pl.bar_plot(data['AGE_PERCENTIL'], data['ICU'], 'ICU Admition by Age ', "Covid-19_ICU_Admission_")
    pl.bar_plot(data['WINDOW'], data['ICU'], 'ICU Admition by Window ', "Covid-19_ICU_Admission_")


def covid_data_preprocessing(data, window=False):
    # get numerical and categorical features.
    categorical = data.select_dtypes(include=['object']).columns
    numerical = data.select_dtypes(exclude=['object']).columns
    # print(categorical)
    # print(numerical)

    # Replace categorigal with numerical variables
    label_encoder = preprocessing.LabelEncoder()
    data[categorical] = data[categorical].apply(label_encoder.fit_transform)
    print("-------------------------------")
    print("Check for missing values: ")
    print("-------------------------------")
    print(data.isnull().sum())
    # Analize missing values:
    # Check record where DISEASE GROUPING is missing
    # inds= np.where(data['DISEASE GROUPING 1'].isnull())
    # data.iloc[inds[0],:]
    # remove records that don't have data in 'DISEASE GROUPING 1'
    data = data.dropna(axis=0, subset=['DISEASE GROUPING 1'])

    # Fill missing values by group using bfill and ffill
    data = data.groupby(['PATIENT_VISIT_IDENTIFIER'], as_index=False).apply(lambda group: group.bfill())
    data = data.groupby(['PATIENT_VISIT_IDENTIFIER'], as_index=False).apply(lambda group: group.ffill())

    # remove data wich has Nan values
    data = data.dropna()

    print("Is any missing values: " + str(data.isna().sum()[1] > 0))
    print("-------------------------------")
    print("Check for outliers.....")
    check_for_outliers(data,'Covid-19_ICU_Admission','covid_outliers')
    print("Check for correlation.....")
    corr_ind = check_for_correlation(data, "Covid-19_ICU_Admission")
    # df = final_data.copy()
    final_data = data.drop(corr_ind, axis=1)
    # Remove patient identifier and window (we don't want to use window to predict icu admission in any stage)
    if window:
        final_data = final_data[final_data['WINDOW'] == 0]

    final_data = final_data.drop(['PATIENT_VISIT_IDENTIFIER', 'WINDOW'], axis=1)

    print('Final data Rows: {} | Final data Columns: {}'.format(final_data.shape[0], final_data.shape[1]))
    # print('Low correlated data Rows: {} | Low correlated data Columns: {}'.format(low_corr_data.shape[0],
    #                                                                               low_corr_data.shape[1]))
    return final_data


def check_for_outliers(data, data_name, folder):
    print("Columns that have outliers: ")
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    ind = IQR[IQR > 0].index
    print(ind)
    print("-------------------------------")
    print("Plot outliers")
    if data_name== 'Covid-19_ICU_Admission':
        for i in range(7, data.shape[1] - 1, 4):
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=data.iloc[:, i:i + 4], orient="h", palette="Set2")

            plt.savefig('./plot_output/{}/{}_{}_boxplot'.format(folder,data_name,str(list(data.iloc[:, i:i + 4].columns))))

        for i in range(0, data.shape[1] - 1):
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=data.iloc[:, -1], y=data.iloc[:, i], data=data)
            plt.savefig('./plot_output/{}/{}_{}_boxplot'.format(folder, data_name, str(data.columns[i])))
    else:
        for i in ind:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=data[i], orient="h", palette="Set2")
            plt.savefig('./plot_output/{}/{}_{}_boxplot'.format(folder, data_name, str(i)))
            plt.clf()

    print("End Plot outliers")
    print("-------------------------------")


def check_for_correlation(data, dataset_name):
    pl.corr_plot(data, dataset_name)
    corr = data.corr()
    corr_data = corr.unstack().sort_values().drop_duplicates()
    ind = corr_data[corr_data.between(0.81, 0.999999)].index
    corr_ind = []
    for i in ind:
        if i == 'WINDOW':
            continue
        corr_ind.append(i[0])
    return corr_ind




def credit_card_analysis(data):
    print("Analise Credit Card Approval  dataset:")
    # Print data shape
    print('Rows: {} | Columns: {}'.format(data.shape[0], data.shape[1]))

    # Replace the '?'s with NaN
    data = data.replace('?', np.nan)
    print("-------------------------------")
    print(" Missing Values Count: {}".format(data.isnull().values.sum()))
    print("-------------------------------")

    missing_ind = data.columns[data.isnull().any()]
    # Replace missing values with most frequent
    for ind in missing_ind:
        if data[ind].dtypes == 'object':
            data[ind] = data[ind].fillna(data[ind].value_counts().index[0])
    #convert second column to int and create bins
    data[1] = data[1].astype(float).astype(int)
    data[1] = pd.cut(x=data[1], bins=range(10, 100, 5), labels=range(15, 100, 5))

    # Check imbalance
    print("Class distribution:  ")
    print("-------------------------------")
    print(data.iloc[:, -1].value_counts())
    print("-------------------------------")


    print("Plotting data ...")
    pl.count_plot(data[15],"Approved", "Not Approved", 'Credit Card Approval', "Credit_Card_Approval_Data")
    pl.bar_plot(data[1], data[15], 'Credit Card Approval by second feature', "Credit_Card_Approval_Data")



def credit_card_preprocessing(data):
    # get numerical and categorical features.
    categorical = data.select_dtypes(include=['object']).columns
    numerical = data.select_dtypes(exclude=['object']).columns
    # print(categorical)
    # print(numerical)

    # Replace categorigal with numerical variables
    label_encoder = preprocessing.LabelEncoder()
    data[categorical] = data[categorical].apply(label_encoder.fit_transform)


    print("-------------------------------")
    print("Check for outliers.....")
    check_for_outliers(data[numerical],'Credit_Card_Approval_Data','credit_card_outliers')
    # Replace outliers
    data[14] = np.where(data[14] > 20000, 2000, data[14])
    data[10] = np.where(data[10] > 15, 12, data[10])
    data[7] = np.where(data[7] > 15, 12, data[7])

    print("Check for correlation.....")
    corr_ind = check_for_correlation(data, "Credit_Card_Approval_Data")
    # drop correlated indexes
    final_data = data.drop(corr_ind, axis=1)
    cols_to_norm = [2, 7, 13, 14]
    final_data[cols_to_norm] = final_data[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))


    print('Final data Rows: {} | Final data Columns: {}'.format(final_data.shape[0], final_data.shape[1]))

    return final_data



