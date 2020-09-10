# # CASE STUDY: BREAST CANCER CLASSIFICATION

# # STEP #1: PROBLEM STATEMENT

# 
# - Predicting if the cancer diagnosis is benign or malignant based on several observations/features 
# - 30 features are used, examples:
#         - radius (mean of distances from center to points on the perimeter)
#         - texture (standard deviation of gray-scale values)
#         - perimeter
#         - area
#         - smoothness (local variation in radius lengths)
#         - compactness (perimeter^2 / area - 1.0)
#         - concavity (severity of concave portions of the contour)
#         - concave points (number of concave portions of the contour)
#         - symmetry 
#         - fractal dimension ("coastline approximation" - 1)
# 
# - Datasets are linearly separable using all 30 input features
# - Number of Instances: 569
# - Class Distribution: 212 Malignant, 357 Benign
# - Target class:
#          - Malignant
#          - Benign
# 
#---------------------------------------------------------------------------------------------------------------------------------
# # STEP #2: IMPORTING DATA

# importing the Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#importing the data
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

#all about the imported data
Keys= cancer.keys()
print(Keys)
data = cancer['data']
target = cancer['target']
target_names = cancer['target_names']
description = cancer['DESCR']
feature_names = cancer['feature_names']
filename = cancer['filename']

#creating the dataset (combining all the independent variables and dependent variables into one table)
dataset = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))

#---------------------------------------------------------------------------------------------------------------------------------
# # STEP #3: VISUALIZING THE DATA

sns.pairplot(dataset, hue = 'target', vars = ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'] )
""" pairplot() = By default, this function will create a grid of Axes such that each numeric variable in data will by shared in the y-axis across a single row and in the x-axis across a single column. The diagonal Axes are treated differently, drawing a plot to show the univariate distribution of the data for the variable in that column.
It is also possible to show a subset of variables or plot different variables on the rows and columns."""

# Blue = 0 = Malignant
# Orange = 1 = Benign

#--   --   --   --   --   --   --   --   --   --   --   --   --   --   --   --   --   --   --   --   --   --   --   --

sns.countplot(dataset['target'])
# seaborn.countplot is a barplot where number of instances of the dependent variable are depicted  

sns.scatterplot(x = 'mean area', y = 'mean smoothness', hue = 'target', data = dataset )

#--   --   --   --   --   --   --   --   --   --   --   --   --
plt.figure(figsize = (20,10)) # for enlarging the heat map
sns.heatmap(dataset.corr(), annot = True)
"""Plot rectangular data as a color-encoded matrix.
This is an Axes-level function and will draw the heatmap into the currently-active Axes if none is provided to the ax argument. Part of this Axes space will be taken and used to plot a colormap"""
#--   --   --   --   --   --   --   --   --   --   --   --   --

#-----------------------------------------------------------------------------------------------------------------------------------
# # STEP #4: MODEL TRAINING (FINDING A PROBLEM SOLUTION)

# getting all the independent variables
X = dataset.iloc[:, 0:30].values
    # alternatively you could also do X = dataset.drop(['target], axis =1)

# getting all the dependent variables
y = dataset.iloc[:, 30].values
    # alternatively you could also do y = dataset['target']

# splitting the data into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =5)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# getting the SVM
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

#creating the support vector classifier object
svc = SVC()
""" C parameter : Controls trade-off between classifying training points correctly and having a smooth decision boundary
    small C (loose) makes cost (penalty) of misclassification low(soft margin)
    large C (strict) makes cost of misclassifiation high (hard margin), forcing the model to explain input data stricter and potentially over fit"""

""" Gamma parameter : controls how far the influence of a single training set reaches
    large gamma : close reach (closer data points have high weight)
    small gamma : far reach ( more generalized solution )"""
    
""" We will apply grid search to choose the right C and Gamma parameters """

# fitting the training model
svc.fit(X_train, y_train)

# Predicting the Test set results
y_pred = svc.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot = True)

#classification report
print(classification_report(y_test, y_pred))

# applying grid search to improve the parameters
param_grid = { 'C' : [0.1, 1, 10, 100], 'gamma' : [1, 0.1, 0.01, 0.001], 'kernel' : ['rbf'] }
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 4)
grid.fit(X_train, y_train)
grid.best_params_
grid_pred = grid.predict(X_test)
cm2 = confusion_matrix(y_test, grid_pred)
sns.heatmap(cm2, annot = True)

#classification report
print(classification_report(y_test, grid_pred))