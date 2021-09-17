#importing libraries (step 1)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
#reading & understanding data (step 2)
df = pd.read_csv('white_wine.csv')
print("Rows, columns: " + str(df.shape))
df.head()
# Missing Values
print(df.isna().sum())
#data exploring
ig = px.histogram(df,x='quality')
fig.show()

#correlation matrix between variables
corr = df.corr()
plt.subplots(figsize=(12,10))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))
plt.show()
X = []
y = []

X = df.iloc[:,1:11].values.astype(float)
y = df.iloc[:,11:12].values.astype(float)
# Normalize feature variables
from sklearn.preprocessing import StandardScaler
X_features = X
X = StandardScaler().fit_transform(X)

# Splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=0)


#logistic regression
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)
model1 = LogisticRegression(random_state=1, multi_class='multinomial', max_iter=100000)
model1.fit(X_train, y_train.ravel())
y_pred1 = model1.predict(X_test)
print(classification_report(y_test, y_pred1))
print('Accuracy of GB classifier on training set: {:.2f}'.format(model1.score(X_train, y_train)))
print('Accuracy of GB classifier on test set: {:.2f}'.format(model1.score(X_test, y_test)))


#SVC
from sklearn.svm import LinearSVC
model2 = LinearSVC(random_state=1, multi_class='ovr', max_iter=100000)
model2.fit(X_train, y_train.ravel())
y_pred2 = model2.predict(X_test)
print(classification_report(y_test, y_pred2))
print('Accuracy of GB classifier on training set: {:.2f}'.format(model2.score(X_train, y_train)))
print('Accuracy of GB classifier on test set: {:.2f}'.format(model2.score(X_test, y_test)))

plt.figure(0)
plt.scatter(y_test, y_pred2, color = 'green')
plt.title('(Support Vector Classification Model)')
plt.xlabel('real wine quality for test set')
plt.ylabel('predicted wine quality for test set')
plt.show()

#GPC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

model3 = GaussianProcessClassifier(random_state=1, multi_class = 'one_vs_rest')
model3.fit(X_train, y_train.ravel()) 
y_pred3 = model3.predict(X_test)
print(classification_report(y_test, y_pred3))

print('Accuracy of GB classifier on training set: {:.2f}'.format(model3.score(X_train, y_train)))
print('Accuracy of GB classifier on test set: {:.2f}'.format(model3.score(X_test, y_test)))

#MLP
from sklearn.neural_network import MLPClassifier
model4 = MLPClassifier(random_state=1,max_iter=100000)
model4.fit(X_train, y_train.ravel())
y_pred4 = model4.predict(X_test)
print(classification_report(y_test, y_pred4))
print('Accuracy of GB classifier on training set: {:.2f}'.format(model4.score(X_train, y_train)))
print('Accuracy of GB classifier on test set: {:.2f}'.format(model4.score(X_test, y_test)))
