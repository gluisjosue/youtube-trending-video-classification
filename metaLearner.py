import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import xgboost as xgb
import pandas as pd
import time

def split_Data(x, y):
	new_x = x.to_numpy()
	xTrain, xTest, yTrain, yTest = train_test_split(new_x, y, test_size =.3)
	xTrain_df = pd.DataFrame(xTrain, columns = x.columns)
	xTest_df = pd.DataFrame(xTest, columns = x.columns)
	yTrain_df = pd.DataFrame(yTrain)
	yTest_df = pd.DataFrame(yTest)
	return xTrain_df, yTrain_df, xTest_df, yTest_df

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
	return pd.DataFrame(clf.predict(xTest), columns = ['category_id'])
	
def compute_f1score(yTrue, yPred):
	print("f1 score (macro) for meta: ", f1_score(yTrue, yPred, average = 'macro'))
	print("f1 score (micro) for meta: ", f1_score(yTrue, yPred, average = 'micro'))
	print("f1 score (weighted) for meta: ", f1_score(yTrue, yPred, average = 'weighted'))

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("--yTest", 
						help = "filename for the support vector machine predict data",
						default = "yTest.csv")
	parser.add_argument("--yPred_svm", 
						help = "filename for the support vector machine predict data",
						default = "yPred_svm.csv")
	parser.add_argument("--yPred_rf", 
						help = "filename for the random forest predict data",
						default = "yPred_rf.csv")
	parser.add_argument("--yPred_knn", 
						help = "filename for the knn predict data",
						default = "yPred_knn.csv")
	parser.add_argument("--yPred_xgb", 
						help = "filename for the xgboost predict data",
						default = "yPred_xgb.csv")
	parser.add_argument("--yPred_dt", 
						help = "filename for the decision tree predict data",
						default = "yPred_dt.csv")
	args = parser.parse_args()
	
	yTest = pd.read_csv(args.yTest)
	yPred_svm = pd.read_csv(args.yPred_svm)
	yPred_rf = pd.read_csv(args.yPred_rf)
	yPred_knn = pd.read_csv(args.yPred_knn)
	yPred_xgb = pd.read_csv(args.yPred_xgb)
	yPred_dt = pd.read_csv(args.yPred_dt)

	predictions = pd.concat([yPred_svm, yPred_rf, yPred_knn, yPred_xgb, yPred_dt], axis = 1)
	predictions.columns = ['yPred_svm', 'yPred_rf', 'yPred_knn', 'yPred_xgb', 'yPred_dt']



	xTrain, yTrain, xTest, yTest = split_Data(predictions, yTest)


	yPred = xg_boost(xTrain, yTrain, xTest, yTest)

	compute_f1score(yTest, yPred)

if __name__ == "__main__":
	main()
