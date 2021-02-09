import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.tree import export_graphviz
from sklearn import metrics
import pandas as pd


def DecisionTrain():
	df = pd.read_csv("vote.tsv", sep='\t')
	##print(data)
	
	X = df.loc[:, df.columns != 'target']
	y = df['target']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) 

	clf = DecisionTreeClassifier(criterion = 'gini')
	##print(len(X_train))

	clf = clf.fit(X_train,y_train)

	y_pred = clf.predict(X_test)

	print("Accuracy with GINI:",metrics.accuracy_score(y_test, y_pred))
	##print(export_text(clf, feature_names=list(df.loc[:, df.columns != 'target'])))
	export_graphviz(clf, feature_names=list(df.loc[:, df.columns != 'target']), out_file="gini.dot")

	clf = DecisionTreeClassifier(criterion = 'entropy')
	clf = clf.fit(X_train,y_train)
	y_pred = clf.predict(X_test)
	print("Accuracy with entropy:",metrics.accuracy_score(y_test, y_pred))
	export_graphviz(clf, feature_names=list(df.loc[:, df.columns != 'target']), out_file="entropy.dot")



DecisionTrain()