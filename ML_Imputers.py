import time
import pandas as pd
import numpy as np
import random
import math
from sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn import preprocessing
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from scipy import stats

#Read dataset
data = pd.read_csv("your file name.csv")

#List of k-neighbours hyperparameter
#list= [5,7,9]
list= [7]

#Create a list of column names (to find out what to dispose) and save it as initial_column_names.txt
def list_column_names():
    column_names = []
    for column in data.columns:
        column_names.append(column)
    with open('initial_column_names.txt', 'w') as f:
        f.write(str(column_names))
list_column_names()

# a list of possible columns that we would like to drop
drop_list = ['##', '##', '##']

#exclude drop_list attributes
data = data.drop(drop_list, axis=1)

#Exclude attributes and create a new clean column names list
def list_column_names_clean():
    column_names_clean = []
    for column in data.columns:
        column_names_clean.append(column)
    result_list = [data,column_names_clean]
    return result_list

data = list_column_names_clean()[0]
column_names_clean = list_column_names_clean()[1]

#See if you have any missing values
#print(data.isnull().sum())

#Keep only no missing rows
no_nans = data[~data.isnull().any(axis=1)]
no_nans.index = range(len(no_nans))


#It is needed because "create_missing_dataframe(percentage)" alters no_nans. But why?
no_nans_dublicate = data[~data.isnull().any(axis=1)]
no_nans_dublicate.index = range(len(no_nans_dublicate))

number = round(0.5 * no_nans.size)

#Count the remaining rows
#print(len(no_nans.index))

def create_missing_dataframe(percentage):
    global number
    number = round(percentage * no_nans.size)
    ix = [(row, col) for row in range(no_nans.shape[0]) for col in range(no_nans.shape[1])]
    for row, col in random.sample(ix, int(round((percentage)*len(ix)))):
        no_nans.iat[row, col] = np.nan
    return no_nans

#Create missing data frame and save it as csv
#data_missing = create_missing_dataframe(0.1)
#data_missing.to_csv("data_missing_10_percent.csv")

#Use one existent missing dataframe
data_missing = pd.read_csv("data_missing_10_percent.csv")

#Drop first column due to csv creation
data_missing = data_missing.iloc[: , 1:]

#Scale data to 0-1. Fit to missing dataset and then aply to original dataset not to leak information
min_max_scaller = preprocessing.MinMaxScaler()
data_missing_scaled = min_max_scaller.fit_transform(data_missing)
data_missing_scaled = pd.DataFrame(data_missing_scaled)
data_missing_scaled.columns = column_names_clean

no_nans_dublicate_scaled = min_max_scaller.transform(no_nans_dublicate)
no_nans_dublicate_scaled = pd.DataFrame(no_nans_dublicate_scaled)
no_nans_dublicate_scaled.columns = column_names_clean


#function to create the full dataset by imputing the mean value in every column
def create_naive_imputed_dataframe():
    naive_imputed_dataset = data_missing.fillna(data_missing.mean())
    return naive_imputed_dataset

#function to create the full dataset by using iterative imputing method (MICE)
def create_iterative_imputed_dataframe():
    lr=LinearRegression(n_jobs=-1)
    impute_iterative = IterativeImputer(estimator=lr, tol=1e-7, max_iter=20, initial_strategy='mean', imputation_order='descending', verbose=4)
    imputed = impute_iterative.fit_transform(data_missing_scaled)
    iterative_imputed_dataset_scaled = pd.DataFrame(imputed, columns=data_missing_scaled.columns)
    iterative_imputed_dataset = min_max_scaller.inverse_transform(iterative_imputed_dataset_scaled)
    iterative_imputed_dataset = pd.DataFrame(iterative_imputed_dataset, columns=data_missing_scaled.columns)
    iterative_imputed_dataset = iterative_imputed_dataset.round(decimals=6)
    iterative_imputed_list_results = [iterative_imputed_dataset_scaled, iterative_imputed_dataset]
    return iterative_imputed_list_results

#function to create the full dataset by using kNN imputing method
def create_kNN_imputed_dataframe(neighbors):
    impute_knn = KNNImputer(n_neighbors=neighbors, weights='distance')
    imputed = impute_knn.fit_transform(data_missing_scaled)
    kNN_imputed_dataset_scaled = pd.DataFrame(imputed, columns=data_missing_scaled.columns)
    kNN_imputed_dataset = min_max_scaller.inverse_transform(kNN_imputed_dataset_scaled)
    kNN_imputed_dataset = pd.DataFrame(kNN_imputed_dataset, columns=data_missing_scaled.columns)
    kNN_imputed_dataset = kNN_imputed_dataset.round(decimals=6)
    kNN_imputed_list_results = [kNN_imputed_dataset_scaled, kNN_imputed_dataset]
    return kNN_imputed_list_results


#Assign variables to functions
#naive_imputed_data = create_naive_imputed_dataframe()
#iterative_imputed_list = create_iterative_imputed_dataframe()
#knn_imputed_list = create_kNN_imputed_dataframe(5)

#Export imputed dataframes to csv
#create_iterative_imputed_dataframe()[1].to_csv("IterativeImputed_10_percent.csv")
#create_kNN_imputed_dataframe(7)[1].to_csv("kNNImputed.csv")
#create_naive_imputed_dataframe()[1].to_csv("naiveImputed_10_percent.csv")

#Check the ks for every imputation method
# Insert the def of method e.g. "create_kNN_imputed_dataframe(neighbors=5)" default = create_naive_imputed_dataframe()
def check_ks_test(method_name):
    print("starting the check_ks_test function")
    pvalues=[]
    for column in column_names_clean:
        pvalue = f"pvalue of column {column} is: {(stats.ks_2samp(method_name[column], no_nans_dublicate_scaled[column]))[1]}"
        pvalues.append(pvalue)
    print(f"The p values of the iterative imputed dataframe are: {pvalues}")
#check_ks_test(naive_imputed_data)

#Espesially for kNN imputing method we do 10 iterations and create a list of the results in order to find the mean of p_value of every column
def check_ks_test_kNN():
    print(f"starting the check_ks_test function for kNN")
    results = []
    for i in list:
        pvalues=[]
        for column in column_names_clean:
            pvalue = f"pvalue of column {column} is: {(stats.ks_2samp(create_kNN_imputed_dataframe(i)[0][column], no_nans_dublicate_scaled[column]))[1]}"
            pvalues.append(pvalue)
            t = time.localtime()
            current_time = time.strftime("%H:%M:%S", t)
            print(f"i appended new pvalue for time {current_time}")
            print(pvalue)
        results.append(pvalues)
    print(results)
    with open('kNN ks test results fo 50 percent.txt', 'w') as f:
        f.write(str(results))
#check_ks_test_kNN()

#Find the average RMSE of the imputed values by first:substract the imputed dataset from inital dataset and then find square, 
# sum imputed values and divide by number of imputed values and then find the square root, do 10 iterations and create a list of the results
# Insert the def of method e.g. "create_iterative_imputed_dataframe()"
def find_RMSE(method):
    temp1 = no_nans_dublicate_scaled.sub(method)
    temp2 = temp1.mul(temp1)
    RMSE = math.sqrt((temp2.sum().sum())/number)
    print(f" The RMSE value of the iterative imputed dataframe is: {RMSE}")
#find_RMSE(create_iterative_imputed_dataframe()[0])

#Espesially for kNN imputing method we do 10 iterations and create a list of the results in order to find the mean of RMSE
def find_RMSE_kNN():
    print(f"starting the RMSE function for kNN")
    results = []
    for i in list:
        temp1 = no_nans_dublicate_scaled.sub(create_kNN_imputed_dataframe(i)[0])
        temp2 = temp1.mul(temp1)
        RMSE = math.sqrt((temp2.sum().sum())/number)
    results.append(RMSE)
    print(results)
    with open('RMSE_kNN_50_percent.txt', 'w') as f:
        f.write(str(results))
#find_RMSE_kNN()

#Find the maximum persentage difference between the imputed values and the original dataset
def find_maxx_diff(method):
    temp1 = no_nans_dublicate.subtract(method)
    temp2 = temp1.abs()
    temp2.to_csv(f"abs_diff_for_iterative_impouted.csv")
    temp3 = temp2.div(no_nans_dublicate)
    temp_series = temp3.max()
    max_per_diff = temp_series.max()
    print(f"The maximum difference is: {max_per_diff}")
#find_maxx_diff(naive_imputed_data)


#Espesially for kNN imputing method we do 10 iterations and create a list of the results in order to find the mean of maxx diff
def find_maxx_diff_kNN():
    print(f"starting the maxx diff kNN function")
    results = []
    for i in list:    
        temp1 = no_nans_dublicate.sub(create_kNN_imputed_dataframe(i)[1])
        temp2 = temp1.abs()
        #temp3 = temp2.div(no_nans_dublicate)
        temp2.to_csv(f"abs_diff_for_k={i}.csv")
        #temp_series = temp3.max()
        #max_per_diff = temp_series.max()
    #results.append(max_per_diff)
#find_maxx_diff_kNN()


