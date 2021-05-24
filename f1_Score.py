import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import argparse
import pandas as pd

def compute_f1score(yTrue, ypred_xgb, ypred_log, ypred_rf, ypred_knn, ypred_nb, ypred_svm, ypred_dt):
	print("f1 score (macro) for xgb: ", f1_score(yTrue, ypred_xgb, average = 'macro'))
	print("f1 score (micro) for xgb: ", f1_score(yTrue, ypred_xgb, average = 'micro'))
	print("f1 score (weighted) for xgb: ", f1_score(yTrue, ypred_xgb, average = 'weighted'))

	print("f1 score (macro) for Logistic Regression: ", f1_score(yTrue, ypred_log, average = 'macro'))
	print("f1 score (micro) for Logistic Regression: ", f1_score(yTrue, ypred_log, average = 'micro'))
	print("f1 score (weighted) for Logistic Regression: ", f1_score(yTrue, ypred_log, average = 'weighted'))

	print("f1 score (macro) for Random Forest: ", f1_score(yTrue, ypred_rf, average = 'macro'))
	print("f1 score (micro) for Random Forest: ", f1_score(yTrue, ypred_rf, average = 'micro'))
	print("f1 score (weighted) for Random Forest: ", f1_score(yTrue, ypred_rf, average = 'weighted'))

	print("f1 score (macro) for knn: ", f1_score(yTrue, ypred_knn, average = 'macro'))
	print("f1 score (micro) for knn: ", f1_score(yTrue, ypred_knn, average = 'micro'))
	print("f1 score (weighted) for knn: ", f1_score(yTrue, ypred_knn, average = 'weighted'))

	print("f1 score (macro) for Naive Bayes: ", f1_score(yTrue, ypred_nb, average = 'macro'))
	print("f1 score (micro) for Naive Bayes: ", f1_score(yTrue, ypred_nb, average = 'micro'))
	print("f1 score (weighted) for Naive Bayes: ", f1_score(yTrue, ypred_nb, average = 'weighted'))

	print("f1 score (macro) for svm: ", f1_score(yTrue, ypred_svm, average = 'macro'))
	print("f1 score (micro) for svm: ", f1_score(yTrue, ypred_svm, average = 'micro'))
	print("f1 score (weighted) for svm: ", f1_score(yTrue, ypred_svm, average = 'weighted'))

	print("f1 score (macro) for decision tree: ", f1_score(yTrue, ypred_dt, average = 'macro'))
	print("f1 score (micro) for decision tree: ", f1_score(yTrue, ypred_dt, average = 'micro'))
	print("f1 score (weighted) for decision tree: ", f1_score(yTrue, ypred_dt, average = 'weighted'))


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--yTest", 
						help = "filename for the support vector machine predict data",
						default = "yTest.csv")
	parser.add_argument("--yPred_svm", 
						help = "filename for the support vector machine predict data",
						default = "yPred_svm.csv")
	parser.add_argument("--yPred_logistic", 
						help = "filename for the Logistic Regression predict data",
						default = "yPred_logistic.csv")
	parser.add_argument("--yPred_rf", 
						help = "filename for the random forest predict data",
						default = "yPred_rf.csv")
	parser.add_argument("--yPred_knn", 
						help = "filename for the knn predict data",
						default = "yPred_knn.csv")
	parser.add_argument("--yPred_nb", 
						help = "filename for the naive bayes predict data",
						default = "yPred_nb.csv")
	parser.add_argument("--yPred_xgb", 
						help = "filename for the xgboost predict data",
						default = "yPred_xgb.csv")
	parser.add_argument("--yPred_dt", 
						help = "filename for the decision tree predict data",
						default = "yPred_dt.csv")
	args = parser.parse_args()

	yTest = pd.read_csv(args.yTest)
	yPred_svm = pd.read_csv(args.yPred_svm)
	yPred_logistic = pd.read_csv(args.yPred_logistic)
	yPred_rf = pd.read_csv(args.yPred_rf)
	yPred_knn = pd.read_csv(args.yPred_knn)
	yPred_nb = pd.read_csv(args.yPred_nb)
	yPred_xgb = pd.read_csv(args.yPred_xgb)
	yPred_dt = pd.read_csv(args.yPred_dt)

	compute_f1score(yTest, yPred_xgb, yPred_logistic, yPred_rf, yPred_knn, yPred_nb, yPred_svm, yPred_dt)


if __name__ == "__main__":
	main()