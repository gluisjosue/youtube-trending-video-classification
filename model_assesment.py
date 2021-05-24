import time
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.naive_bayes import ComplementNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def normalize_data_standard(df):
	x = df.to_numpy()
	scaler = preprocessing.StandardScaler()
	df_scaled = scaler.fit_transform(df)
	df2 = pd.DataFrame(df_scaled, columns = df.columns)
	return df2

def normalize_data_min_max(df):
	x = df.to_numpy()
	scaler = preprocessing.MinMaxScaler()
	df_scaled = scaler.fit_transform(df)
	df2 = pd.DataFrame(df_scaled, columns = df.columns)
	return df2

def n_b_complement(xTrain, yTrain, xTest, yTest):
	clf = ComplementNB()
	clf.fit(xTrain, yTrain)
	print("Running Complement Naive Bayes")
	s = time.time()
	clf.fit(xTrain, yTrain.to_numpy().ravel())
	print("score on the Test: ", clf.score(xTest, yTest))
	e = time.time()
	t = e-s
	print("Elapsed time: ", t)
	return(pd.DataFrame(clf.predict(xTest), columns = ['category_id']))
def support_vec_machine(xTrain, yTrain, xTest, yTest):
	clf = SVC()
	print("Running Support Vector Machine Classifier")
	s = time.time()
	clf.fit(xTrain, yTrain.to_numpy().ravel())
	print("Test")
	print("score on the Test: ", clf.score(xTest, yTest))
	e = time.time()
	t = e-s
	print("Elapsed time: ", t)
	return(pd.DataFrame(clf.predict(xTest), columns = ['category_id']))

def logistic(xTrain, yTrain, xTest, yTest):
	clf = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial', max_iter = 1000)
	s = time.time()
	print("Running Logistic Regression")	
	clf.fit(xTrain, yTrain.to_numpy().ravel())
	print("Test")
	print("score on the Test: ", clf.score(xTest, yTest))
	e = time.time()
	t = e-s
	print("Elapsed time: ", t)
	return(pd.DataFrame(clf.predict(xTest), columns = ['category_id']))

def rf(xTrain, yTrain, xTest, yTest):
	clf = RandomForestClassifier(n_estimators = 20)
	print("Running Random Forest")
	s = time.time()
	clf.fit(xTrain, yTrain.to_numpy().ravel())
	print("Test")
	print("score on the Test: ", clf.score(xTest, yTest))
	e = time.time()
	t = e-s
	print("Elapsed time: ", t)
	return(pd.DataFrame(clf.predict(xTest), columns = ['category_id']))

def knn(xTrain, yTrain, xTest, yTest, k):
	clf = KNeighborsClassifier(n_neighbors = k)
	print("Running knn")
	s = time.time()
	clf.fit(xTrain, yTrain.to_numpy().ravel())
	print("Test")
	print("score on the Test: ", clf.score(xTest, yTest))
	e = time.time()
	t = e-s
	print("Elapsed time: ", t)
	return(pd.DataFrame(clf.predict(xTest), columns = ['category_id']))

def xg_boost(xTrain, yTrain, xTest, yTest):
	clf = xgb.XGBClassifier(max_depth= 10, learning_rate = 0.5, n_estimators= 11)
	print("Running xgboost")
	s = time.time()
	clf.fit(xTrain, yTrain.to_numpy().ravel())
	print("Test")
	print("score on the Test: ", clf.score(xTest, yTest))
	e = time.time()
	t = e-s
	print("Elapsed time: ", t)
	return(pd.DataFrame(clf.predict(xTest), columns = ['category_id']))	

def decision_tree_classifier(xTrain, yTrain, xTest, yTest):
	clf = DecisionTreeClassifier()
	print("Running Decision Tree")
	s = time.time()
	clf.fit(xTrain, yTrain.to_numpy().ravel())
	print("Test")
	print("score on the Test: ", clf.score(xTest, yTest))
	e = time.time()
	t = e-s
	print("Elapsed Time: ", t)
	return(pd.DataFrame(clf.predict(xTest), columns = ['category_id']))

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("--xTrain",
						help = 'filename for the xTrain data',
						default = "xTrain.csv")
	parser.add_argument("--yTrain",
						help = 'filename for the yTrain data',
						default = "yTrain.csv")
	parser.add_argument("--xTest",
						help = 'filename for the xTest data',
						default = "xTest.csv")
	parser.add_argument("--yTest",
						help = 'filename for the yTest data',
						default = "yTest.csv")

	parser.add_argument("yPred_svm", 
						help = "filename for the support vector machine predict data")
	parser.add_argument("yPred_logistic", 
	 					help = "filename for the Logistic Regression predict data")
	parser.add_argument("yPred_rf", 
	 					help = "filename for the random forest predict data")
	parser.add_argument("yPred_knn", 
	 					help = "filename for the knn predict data")
	parser.add_argument("yPred_nb", 
	 					help = "filename for the naive bayes predict data")
	parser.add_argument("yPred_xgb", 
	 					help = "filename for the xgboost predict data")
	parser.add_argument("yPred_dt", 
						help = "filename for the decision tree predict data")	
	args = parser.parse_args()

	xTrain = pd.read_csv(args.xTrain)
	yTrain = pd.read_csv(args.yTrain)
	xTest = pd.read_csv(args.xTest)
	yTest = pd.read_csv(args.yTest)

	xTrain_normalized_standard = normalize_data_standard(xTrain)
	xTest_normalized_standard = normalize_data_standard(xTest)
	xTrain_normalized_min_max = normalize_data_min_max(xTrain)
	xTest_normalized_min_max = normalize_data_min_max(xTest)

	ypred_svm = support_vec_machine(xTrain_normalized_standard, yTrain, xTest_normalized_standard, yTest)
	ypred_log = logistic(xTrain_normalized_standard, yTrain, xTest_normalized_standard, yTest)
	ypred_rf = rf(xTrain_normalized_standard, yTrain, xTest_normalized_standard, yTest)
	ypred_knn = knn(xTrain_normalized_standard, yTrain, xTest_normalized_standard, yTest, 1)
	ypred_nb = n_b_complement(xTrain_normalized_min_max, yTrain, xTest_normalized_min_max, yTest)
	ypred_xgb = xg_boost(xTrain_normalized_standard, yTrain, xTest_normalized_standard, yTest)
	ypred_dt = decision_tree_classifier(xTrain_normalized_standard, yTrain, xTest_normalized_standard, yTest)
	

	ypred_svm.to_csv(args.yPred_svm, index = False)
	ypred_log.to_csv(args.yPred_logistic, index = False)
	ypred_rf.to_csv(args.yPred_rf, index = False)
	ypred_knn.to_csv(args.yPred_knn, index = False)
	ypred_nb.to_csv(args.yPred_nb, index = False)
	ypred_xgb.to_csv(args.yPred_xgb, index = False)
	ypred_dt.to_csv(args.yPred_dt, index = False)
if __name__ == "__main__":
	main()