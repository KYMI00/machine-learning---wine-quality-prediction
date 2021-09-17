#importing libraries (step 1)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

#reading & understanding data (step 2)
df = pd.read_csv('white_wine.csv')
X = df.iloc[:,1:11].values.astype(float)
y = df.iloc[:,11:12].values.astype(float)

# Normalize feature variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

#Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=0)

#SVR
from sklearn.metrics import classification_report
from sklearn.svm import SVR
print('Support Vector Regression')
reg1 = SVR(kernel='rbf')
reg1.fit(X_train,y_train.ravel())
y_pred1 = reg1.predict(X_test)
y_pred1 = np.round(y_pred1)
print('Accuracy of svr classifier on training set: {:.2f}'.format(reg1.score(X_train, y_train)))
print('Accuracy of svr classifier on test set: {:.2f}'.format(reg1.score(X_test, y_test)))


#GPR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
kernel = DotProduct() + WhiteKernel()
reg2 = GaussianProcessRegressor(kernel)
reg2.fit(X_train,y_train.ravel())
y_pred2 = reg2.predict(X_test)
y_pred2 = np.round(y_pred2)
#print(str(y_test))

print('Accuracy of GPR classifier on training set: {:.2f}'.format(reg2.score(X_train, y_train)))
print('Accuracy of GPR classifier on test set: {:.2f}'.format(reg2.score(X_test, y_test)))

#plt.figure(1)
#plt.scatter(y_test, y_pred2, color = 'green')
#plt.title('(Gaussian Process Regression Model)')
#plt.xlabel('true wine quality for test set')
#plt.ylabel('predicted wine quality for test set')
#plt.show()

#MLP
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
reg3 = MLPRegressor(hidden_layer_sizes=(64,64,64),activation="relu" ,random_state=1, max_iter=2000)
reg3.fit(X_train,y_train.ravel())
y_pred3 = reg3.predict(X_test)
y_pred3 = np.round(y_pred3)

#plt.figure(2)
#plt.scatter(y_test,y_pred3, color = 'green')
#plt.title('(Multilayer perceptron Model)')
#plt.xlabel('true wine quality for test set')
#plt.ylabel('predicted wine quality for test set')
#plt.show()