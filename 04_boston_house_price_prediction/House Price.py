# Ajay Choudhury (IISERB, EECS Dept. Roll- 18018)
# To Select various models refer to line 70 and uncomment only the model required

# <----------------- Using Regression Models----------------->

# Importing required libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Taking the dataset 'data.csv' as input
housing = pd.read_csv("data.csv")

# print(housing.head())                             # Gives a preview of the dataset
# print(housing.info())                             # Check for any null values
# print(housing['CHAS'].value_counts())             # Count of values for understanding the data
# print(housing.describe())                         # Understanding the properties like mean, std, max, etc.

# For plotting histogram of each features in dataset
housing.hist(bins=50, figsize=(20, 15))
plt.show()

# Splitting the dataset for Training and Testing
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# print(strat_test_set['CHAS'].value_counts())      # Count of values of each type in test and training set
# print(strat_train_set['CHAS'].value_counts())

# Training set is copied to housing variable
housing = strat_train_set.copy()

# Looking for correlations of features with the final price
# corr_matrix = housing.corr()
# print(corr_matrix['MEDV'].sort_values(ascending=False))

# Plotting the correlations of some important features with each other
from pandas.plotting import scatter_matrix
attributes = ["MEDV", "RM", "ZN", "LSTAT"]
scatter_matrix(housing[attributes], figsize = (12,8))
plt.show()

# Removing the Labels from the training dataset
housing = strat_train_set.drop("MEDV", axis=1)      # Dropping the last column i.e. Predicted Price
housing_labels = strat_train_set["MEDV"].copy()     # Only the column of Predicted Price is included

# Creating the Pipeline
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

# Passing the training dataset through the Pipeline
housing_num_tr = my_pipeline.fit_transform(housing)
# print(housing_num_tr.shape)                       # It is (404, 13) i.e. the desired value

# Selecting the model
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

# model = LinearRegression()
# model = DecisionTreeRegressor()
# model = SVR()
model = RandomForestRegressor()                         # Lowest RMSE Model
# model = KNeighborsRegressor()
model.fit(housing_num_tr, housing_labels)

# Testing the model on a few rows of Training dataset
some_data = housing.iloc[:5]                            # 5 rows of the training dataset
some_labels = housing_labels.iloc[:5]
prepared_data = my_pipeline.transform(some_data)        # Passing the prepared data through the pipeline
# print(prepared_data)
# print("Predicted Price of 5 Rows of Training dataset:")
# print(model.predict(prepared_data), "\n")
# print("Original Price:")
# print(list(some_labels))

# Evaluating the model using Cross-validation
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)

# Function to print the RMSE scores along with Mean and Standard deviation of the RMSE scores
def print_scores(scores):
    print("Here is the Evaluation of the applied model:\n")
    print("RMSE Scores:", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std(), "\n")

# Calling the function to print the scores
print_scores(rmse_scores)

# Testing the model on Test dataset
X_test = strat_test_set.drop("MEDV", axis=1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

# Printing the final predictions on test dataset
print("Predicted Prices on Test dataset:")
print(final_predictions, "\n")
print("Original Prices:")
print(list(Y_test),"\n")
print("The RMSE of the model on Test dataset:")
print(final_rmse, "\n")

# # Copy the array values printed for testing the model below
# print(prepared_data[0])


# Using the model----> Take a input of feature values
# Here the values are not actual but are passed through the pipeline
features = np.array([[-0.43942006,  3.12628155, -1.12165014, -0.27288841, -1.42262747,
       -0.23979304, -1.31238772,  2.61111401, -1.0016859 , -0.5778192 ,
       -0.97491834,  0.41164221, -0.86091034]])
print("Predicted Price is: ", model.predict(features))
print("Original Price is: ", list(some_labels[:1]), "\n")





# <----------------- RMSE of the Models ----------------->

# 1. Linear Regression
# Mean of RMSE: 5.030437102767304
# Standard deviation: 1.0607661158294832

# 2. Decision Tree
# Mean of RMSE: 4.218344806994929
# Standard deviation: 0.7928741428323183

# 3. SVR
# Mean of RMSE: 5.606712889302848
# Standard deviation: 1.70961043370698

# 4. Random Forest Regressor                                      # Best Results Produced as RMSE is lowest
# Mean of RMSE: 3.3597803308307173
# Standard deviation: 0.7295102580579761

# 5. K-Neighbors Regressor
# Mean of RMSE: 4.740008476062452
# Standard deviation: 1.4262406795924745

# 6. Neural Network
# RMSE: 4.060599385963832

# <----------------- RMSE of the Models ----------------->





# <----------------- Using Feed Forward Neural Network----------------->

import tensorflow as tf
from sklearn import preprocessing

# Normalising the values of the features
housing = preprocessing.normalize(housing)
X_test = preprocessing.normalize(X_test)

# Taking 5 Rows from the training dataset for evaluation
ann_data = housing[:5]
ann_labels = housing_labels[:5]

# Creating the Neural Network Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
def HousePricePredictionModel():
    model = Sequential()
    model.add(Dense(128,activation='relu',input_shape=(housing[0].shape)))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
    return model

k = 4
num_val_samples = len(housing)
num_epochs = 50
all_scores = []

model = HousePricePredictionModel()
history = model.fit(x=housing, y=housing_labels, epochs=num_epochs, batch_size=1, verbose=1, validation_data=(X_test,Y_test))

# # Testing the model on 5 Rows of Training dataset
# print(model.predict(ann_data))
# print(list(ann_labels))

# Evaluating the model
housing_predictions = model.predict(housing)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)
print("\nRMSE of the FF Neural network model: ", rmse, "\n")

# Testing the model on Test dataset
final_predictions = model.predict(X_test)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

# Printing the final predictions on test dataset
print("Predicted Prices on Test dataset:")
print(final_predictions, "\n")
print("Original Prices:")
print(list(Y_test),"\n")
print("The RMSE of the FF Neural network model on Test dataset:")
print(final_rmse, "\n")

# # Copy the array values printed for testing the model below
# print(ann_data[0])

# Using the model----> Take a input of feature values
# Here the values are not actual but are normalised beforehand
features = np.array([[9.42440546e-05, 1.56454127e-01, 7.11866277e-03, 0.00000000e+00,
       7.66625221e-04, 1.19452726e-02, 6.25816507e-02, 1.80319248e-02,
       1.95567658e-03, 6.16038124e-01, 3.20730960e-02, 7.68365773e-01,
       1.28487952e-02]])
print("Predicted Price is: ", model.predict(features))
print("Original Price is: ", list(ann_labels[:1]), "\n")





# <----------------- Using Unsupervised Learning ----------------->
# Using unsupervised learning to get a cluster of prices based on the nitric oxides concentration (parts per 10 million) in the surroundngs

# Importing libraries
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Loading the dataset
housing_uns = pd.read_csv("data.csv")

# # Plotting a relation to see if clustering is meaningful
plt.scatter(housing_uns['NOX'], housing_uns['MEDV'])
plt.show()

# Implementing MinMaxScaler to transform the data
scaler = MinMaxScaler()
housing_uns['MEDV'] = scaler.fit_transform(housing_uns[['MEDV']])
housing_uns['NOX'] = scaler.fit_transform(housing_uns[['NOX']])

# Deciding the number of clusters
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(housing_uns[['NOX', 'MEDV']])

housing_uns['cluster'] = y_predicted
# print("The cluster centers are: ", km.cluster_centers_, "\n")                    # The centres of clusters

df1 = housing_uns[housing_uns.cluster==0]
df2 = housing_uns[housing_uns.cluster==1]
df3 = housing_uns[housing_uns.cluster==2]

plt.scatter(df1.NOX, df1['MEDV'], color='green')
plt.scatter(df2.NOX, df2['MEDV'], color='red')
plt.scatter(df3.NOX, df3['MEDV'], color='yellow')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], color='purple', marker='*', label='centroid')

plt.xlabel('NOX')
plt.ylabel('MEDV')
plt.legend()
plt.show()

print("Printed the scatter plot of K-Mean Clustering of Price based on NOX feature.\n")

# To plot the relation between sum of squared errors and K 
k_rng = range(1,10)
sse = []

for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(housing_uns[['NOX', 'MEDV']])
    sse.append(km.inertia_)

print("The array of SSE: ", sse, "\n")

plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)
plt.show()





# <----------------- Using Logistic Regression ----------------->
# Using logistic regression and a modified dataset to predict whether the price of the house is above $30K(1) or below $30K(0)

# Importing required libraries
from pandas.core.frame import DataFrame
from pandas.core.indexes.base import Index
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Loading the dataset
housing_log = pd.read_csv("data_log.csv")  

# Prepare the training set
X = housing_log.iloc[:, :-1]                   # Excluding the last column which contains the binary classification
y = housing_log.iloc[:, -1]                    # Only including the last column as labels

# Splitting the dataset itself to test the accuracy
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=42)

# Training the model
model = LogisticRegression()
model.fit(x_train, y_train)

# Testing the model
predictions = model.predict(x_test)

# Checking the precision, recall and f1-score
print("\nClassification report:\n", classification_report(y_test, predictions), "\n")
print("Accuracy of the Model: ", accuracy_score(y_test, predictions), "\n")

# Printing the predictions for test dataset
print("The predictions for the Test dataset are: ", predictions, "\n")

# Enter features to check whether the priice is below $30K(0) or above $30K(1)
# Copy the original values of features from the dataset for testing
features = np.array([[0.7842,0,8.14,0,0.538,5.99,81.7,4.2579,4,307,21,386.75,14.67]])
print("Predicted category is (0 for below $30K and 1 for above $30K): ", model.predict(features))