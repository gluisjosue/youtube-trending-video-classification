import re
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn import preprocessing



def heat_map(df, y):
	newdf = pd.concat([df, y], axis = 1)
	correlation_matrix = newdf.corr(method = 'pearson')
	ax = sns.heatmap(correlation_matrix)
	import matplotlib.pyplot as plt
	plt.show()

def extract_label(df):
	label = df['category_id']
	y = pd.DataFrame(label)
	df2 = df.drop(columns = ['category_id'])
	return df2, y


def split_Data(x, y):
	new_x = x.to_numpy()
	xTrain, xTest, yTrain, yTest = train_test_split(new_x, y, test_size =.3)
	xTrain_df = pd.DataFrame(xTrain, columns = x.columns)
	xTest_df = pd.DataFrame(xTest, columns = x.columns)
	yTrain_df = pd.DataFrame(yTrain)
	yTest_df = pd.DataFrame(yTest)
	return xTrain_df, yTrain_df, xTest_df, yTest_df

def sentiment_analysis(df):
	x = df.to_numpy()
	list_of_subjectivity= []
	list_of_polarity = []
	for item in x:
		string = TextBlob(item)
		list_of_subjectivity.append(string.sentiment.subjectivity)
		list_of_polarity.append(string.sentiment.polarity)
	df2 = pd.DataFrame(data = list_of_subjectivity)
	df3 = pd.DataFrame(data = list_of_polarity)
	return df2, df3

def binary_extraction(df):
	binary_list = []
	x = df.to_numpy()
	for item in x:
		if item ==True:
			binary_list.append(1)
		else:
			binary_list.append(0)
	df2 = pd.DataFrame(data = binary_list)
	return df2

def label_encoder(df):
	x = df.to_numpy()
	le = preprocessing.LabelEncoder()
	new_x = le.fit_transform(x)
	df2 = pd.DataFrame(data = new_x)

def word_count(df, separator):
	word_count_list = []

	x = df.to_numpy()
	for item in x:
		wordList = item.split(separator)
		word_count_list.append(len(wordList))

	df2 = pd.DataFrame(data = word_count_list)
	return df2

def check_capitalization(df):
	cap_list = []
	x = df.to_numpy()
	for item in x:
		alphanumeric_sentence = re.sub(r'\W+', '', item)
		if alphanumeric_sentence.isupper():
			cap_list.append(1)
		else:
			cap_list.append(0)
	df2 = pd.DataFrame(data = cap_list)
	return df2

def find_likes_dislikes_ratio(df1, df2):
	likes_ratio = []
	dislikes_ratio = []
	x = df1.to_numpy()
	y = df2.to_numpy()

	for item1, item2 in zip(x, y):
		total = item1+item2
		if total ==0:
			likes_ratio.append(0)
			dislikes_ratio.append(0)
		else:
			r = item1/total
			r2 = item2/total
			likes_ratio.append(r)
			dislikes_ratio.append(r2)

	df3 = pd.DataFrame(data = likes_ratio)
	df4 = pd.DataFrame(data = dislikes_ratio)
	return df3, df4

def find_views_ratio(df1, views):
	ratio = []
	x = df1.to_numpy()
	y = views.to_numpy()

	for item1, view in zip(x, y):
		if item1==0:
			ratio.append(0)
		else:
			r = item1/view
			ratio.append(r)
	df2 = pd.DataFrame(data = ratio)
	return df2

def featureCreation(df):
	df['description']= df['description'].fillna("")
	df['comments_disabled'] = binary_extraction(df['comments_disabled'])
	df['ratings_disabled'] = binary_extraction(df['ratings_disabled'])
	df['video_error_or_removed'] = binary_extraction(df['video_error_or_removed'])
	df['publish_time'] = pd.to_datetime(df['publish_time'], infer_datetime_format = True)
	df['publish_hour'] = df.publish_time.dt.hour
	df['publish_hour'] = pd.to_numeric(df['publish_hour'])
	df['publish_day'] = df.publish_time.dt.day
	df['publish_day'] = pd.to_numeric(df['publish_day'])
	df['publish_month'] = df.publish_time.dt.month
	df['publish_month'] = pd.to_numeric(df['publish_month'])
	df['publish_year'] = df.publish_time.dt.year
	df['publish_year'] = pd.to_numeric(df['publish_year'])
	df['title subjectivity'], df['title polarity'] = sentiment_analysis(df['title'])
	df['tags subjectivity'], df['tags polarity'] = sentiment_analysis(df['tags'])
	df['description subjectivity'], df['description polarity'] = sentiment_analysis(df['description'])
	df['channel_title'] = label_encoder(df['channel_title'])
	df['description_count'] = word_count(df['description'], " ")
	df['title_word_count'] = word_count(df['title'], " ")
	df['tags_count'] = word_count(df['tags'], "|")
	df['title_capitalization'] = check_capitalization(df['title'])
	df['likes_dislikes_ratio'], df['dislikes_likes_ratio'] = find_likes_dislikes_ratio(df['likes'], df['dislikes'])
	df['likes_views_ratio'] = find_views_ratio(df['likes'], df['views'])
	df['dislikes_views_ratio'] = find_views_ratio(df['dislikes'], df['views'])
	df= df.drop(columns = ['video_id', 'trending_date', 'thumbnail_link', 'description', 'title', 'tags', 'channel_title', 'publish_time'])
	return df
	
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--USvideos",
						help = "filename for the US videos",
						default = "USvideos.csv")
	parser.add_argument("--GBvideos",
						help = "filename for the Great Britain videos",
						default = "GBvideos.csv")
	parser.add_argument("xTrain",
						help = 'filename for the xTrain data output file')
	parser.add_argument("yTrain",
						help = 'filename for the yTrain data output file')
	parser.add_argument("xTest",
						help = 'filename for the xTest data output file')
	parser.add_argument("yTest",
						help = 'filename for the yTest data output file')
	args = parser.parse_args()

	#READING IN FILES FROM DIRECTORY
	US_videos = pd.read_csv(args.USvideos)
	GB_videos = pd.read_csv(args.GBvideos)
	videos = pd.concat([US_videos, GB_videos], axis = 0)
	videos = videos.sample(frac=1).reset_index(drop=True)

	xFeat, y= extract_label(videos)
	xFeat= featureCreation(xFeat)
	heat_map(xFeat, y)
	xTrain, yTrain, xTest, yTest = split_Data(xFeat, y)

	xTrain.to_csv(args.xTrain, index = False)
	yTrain.to_csv(args.yTrain, index = False)
	xTest.to_csv(args.xTest, index = False)
	yTest.to_csv(args.yTest, index = False)	

if __name__ == "__main__":
	main()