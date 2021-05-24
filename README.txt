Overview of python files

data_creation.py does the following:
take in the regional video data
Extract Label
Perform feature extraction
Perform feature selection
Create xTrain.csv. yTrain.csv, xTest.csv, yTest.csv

model_assesment.py does the following:
takes in xTrain, yTrain, xTest, yTest
predict the label for the selected models
score the models using model.score(xTest) function
Creates a .csv file for every model prediction

f1_score.py does the following:
takes in every prediction along with the true label
computes the f1 score for macro, micro, and weighted averages

metaLearner.py does the following: 
takes in every prediction along with the true label
combines it all into one dataframe
performs XGBOOST classifier to aquire the final result.
