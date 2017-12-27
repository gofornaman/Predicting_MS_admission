import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import sklearn
import io
import requests

# ----------------------------------
#	STEP 1 - READ THE DATASET AND UNDERSTAND 

dataset = pd.read_csv("binary.csv")

print(type(dataset))

#looking into head
print(dataset.head())

#no of rows, cols
print(dataset.shape)

#info bout emm data
print(dataset.info())

#figure out non NA values
print(dataset.count())

#info bout cols
print(dataset.columns)

#lets summarize the dataset

#get sum
print(dataset.sum())

#get stats
print(dataset.describe())

#get mean
print(dataset.mean())

#get median
print(dataset.median())

print("#----------------------xx-------------------------")

print('#	STEP 2 VISUALISATION  	#')

import seaborn as sns
sns.set(style="dark",context="talk")
f,(ax1, ax2) = plt.subplots(1,2, figsize=(10,6))

#gre scores
sns.distplot(dataset.ix[:,1], ax=ax1, color="b");

#gpa scores
sns.distplot(dataset.ix[:,2], ax=ax2, color="g");

plt.show()

#lets create multivariate plots
sns.pairplot(dataset, hue='admit',palette="husl", x_vars=["gre","gpa","rank"], y_vars=["gre","gpa","rank"], size=4)
plt.show()


from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


#convert dataframe into matrix
dataArray = dataset.values

#splitting input features & o/p vars
X = dataArray[:,1:4]
y = dataArray[:,0:1]

#splitting training & testing
validation_size = 0.10
seed = 9
X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=validation_size, random_state = seed)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


print('#----------------------xx------------------------------#')
print('---------SEC 3 MODELING--------------')

#models - LR,LDA,KNN, CART, RF, NB, SVM

num_trees = 200
max_features = 3
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('RF', RandomForestClassifier(n_estimators=num_trees, max_features=max_features)))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))


#fit models & eval

results = []
names = []
scoring = 'accuracy'

#bring out em cross validation
for name, model in models:
	kfold = KFold(n_splits = 10, random_state=7)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring = scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name,cv_results.mean(), cv_results.std())
	print(msg)


#lets box plot model scores

fig = pyplot.figure()
fig.suptitle('ML algo comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

#create prediction model
model = LogisticRegression()

#fit model
model.fit(X_train, Y_train)

#predict!
predictions = model.predict(X_test)

#check accuracy
print("Model --- LogisticRegression")
print("Accuracy: {} ".format(accuracy_score(Y_test,predictions) * 100)
print(classification_report(Y_test, predictions))


#plotting confusion matrix on heatmap

cm = confusion_matrix(Y_test, predictions)
sns.heatmap(cm, annot=True, xticklabels=['reject','admit'], yticklabels=['reject','admit'])
plt.figure(figsize=(3,3))
plt.show()



#make predictions on some new data
#like a boss

new_data = [(720,4,1), (300,2,3) , (400,3,4) ]

#conv to numpy arr
new_array = np.asarray(new_data)

#o/p labels
labels=["reject","admit"]

#make pred
prediction=model.predict(new_array)

#get no of test cases used
no_of_test_cases, cols = new_array.shape

#res
for i in range(no_of_test_cases):
	print("Status of Student with GRE scores = {}, GPA grade = {}, Rank = {} will be ----- {}".format(new_data[i][0],new_data[i][1],new_data[i][2], labels[int(prediction[i])]))




print('#--------------END------THANKS-------------#')