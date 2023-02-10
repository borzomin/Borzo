import _pickle as cPickle
import argparse
import csv
import itertools
import os

import matplotlib.pyplot as plt
import numpy
import pandas
from nltk.corpus import stopwords
from sklearn.cross_validation import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from textblob import TextBlob

FILEPATH = os.path.dirname(os.path.abspath(__file__))

# Reading the Data Set
DATASET = pandas.read_csv(FILEPATH + '/DataSet/SMSSpamCollection', sep='\t', quoting=csv.QUOTE_NONE, names=["label", "message"])
MODELFILEPOSTFIX = '_model'


# ************************************* #
#                                       #
#        Preprocessing Functions        #
#                                       #
# ************************************* #

def splitToLemmas_NoStopWord(message):
	'''
	convert words to base form (lemma), also removes Step words (common words)

	:param str message :
	:return: list of word lemmas
	:rtype: list
	'''
	StopWords = set(stopwords.words('english'))
	message = bytes(message, 'utf-8').decode().lower()  # convert to UTF-8, lowercase is for normalization
	words = TextBlob(message).words
	# use base form (lemma) of each word instead of the word itself if the word is not stop word and return list of them.
	return [word.lemma for word in words if word not in StopWords]


def splitToLemmas(message):
	'''
	convert words to  base form (lemma)

	:param str message :
	:return: list of word lemmas
	:rtype: list
	'''
	message = bytes(message, 'utf-8').decode().lower()  # convert to UTF-8, lowercase is for normalization
	words = TextBlob(message).words
	# use base form (lemma) of each word instead of the word itself and return list of them.
	return [word.lemma for word in words]


def splitToTokens(message):
	'''
	converts message to list of words

	:param str message:
	:return: list of words
	:rtype: list
	'''
	message = bytes(message, 'utf-8').decode()
	return TextBlob(message).words


# ************************************* #
#                                       #
#           Utility Functions           #
#                                       #
# ************************************* #

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, percent=False):
	'''
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	from `Confusion matrix <http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html>`_ documentations.

	:param cm: confusion matrix
	:param list classes: Labels of matrix
	:param bool normalize: normalize data or not
	:param str title: title of graph
	:param cmap: color map
	:param percent: If normalize is True, show percent of result or not
	:return: None
	'''
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis] * (100 if percent is True else 1)
		print("Normalized confusion matrix:\n")
	else:
		print('Confusion matrix, without normalization:\n')

	print(cm)

	plt.close()
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = numpy.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
		         horizontalalignment="center",
		         color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.show()


def showModelDetail(detector, test_set_data, test_set_label, title=None):
	print("\n\n************************\t" + title + "\t************************\n")
	# Print result of tran using cross validation
	predictions = detector.predict(test_set_data)
	print("\t========\tConfusion Matrix : Classification Accuracy\t========\n")

	cm = confusion_matrix(test_set_label, predictions)

	plot_confusion_matrix(cm, ["ham", "spam"], title=title + ' Confusion matrix, without normalization')
	plot_confusion_matrix(cm, ["ham", "spam"], title=title + ' Normalized confusion matrix', normalize=True)

	print("\t================\tClassification Report\t\t================\n")

	print(classification_report(test_set_label, predictions))


def saveModelToFile(model, file_name):
	'''
	Saves a model to file using cPickle for future use

	:param object model: model to save
	:param str file_name:
	:return: True on success and False n failure
	:rtype bool
	'''
	try:
		directory = FILEPATH + '/models/'

		if not os.path.exists(directory):
			os.makedirs(directory)

		with open(directory + file_name, 'wb') as out_file:
			cPickle.dump(model, out_file)
		print('model written to: ' + file_name)
		return True
	except Exception as e:
		print("ERROR Writing Model To File:\n", e)
		return False


def optimizeParameters(data_set, classifier, parameters, title=None):
	'''
	Optimize given classifier kernel's parameters using Grid Search Cross Validation and return it
	:param array-like data_set:
	:param classifier: classifier to use in pipeline
	:param parameters: parameters to optimize with Grid Search Cross Validation
	:return: trained classifier
	'''
	# split data set for cross validation
	msg_train, msg_test, label_train, label_test = train_test_split(data_set['message'], data_set['label'], test_size=0.2)

	# create pipeline
	pipeline = Pipeline([
		('bow', CountVectorizer()),  # strings to token integer counts
		('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF (Term Frequency - Inverse Document Frequency) scores
		('classifier', classifier)  # train on TF-IDF vectors with given Classifier
	])

	# pipeline parameters to automatically explore and tune
	grid = GridSearchCV(
		pipeline,
		param_grid=parameters,  # parameters to tune via cross validation
		refit=True,  # fit using all available data at the end, on the best found param combination
		n_jobs=-1,  # number of cores to use for parallelization; -1 for "all cores"
		scoring='accuracy',  # what score are we optimizing?
		cv=StratifiedKFold(label_train, n_folds=5),  # what type of cross validation to use
	)

	detector = grid.fit(msg_train, label_train)  # find the best combination of parameters for this data

	# Print result of tran using cross validation
	showModelDetail(detector, msg_test, label_test, title)

	return detector


def predict(message):
	'''
	restore models from file and Test given message with them, then return result
	:param str message:
	:return: (svm result, nb result, knn result)
	:rtype:tuple
	'''
	nb_detector = cPickle.load(open(FILEPATH + '/models/nb' + MODELFILEPOSTFIX + '.pkl', mode='rb'))
	svm_detector = cPickle.load(open(FILEPATH + '/models/svm' + MODELFILEPOSTFIX + '.pkl', mode='rb'))
	knn_detector = cPickle.load(open(FILEPATH + '/models/knn' + MODELFILEPOSTFIX + '.pkl', mode='rb'))

	nb_predict = nb_detector.predict([message])[0]
	svm_predict = svm_detector.predict([message])[0]
	knn_predict = knn_detector.predict([message])[0]

	return svm_predict, nb_predict, knn_predict


def showDeatils(data_set):
	# print each label detail including count, unique, most frequent
	print(data_set.groupby('label').describe(), '\n')

	# Set length property for each message
	data_set['length'] = data_set['message'].map(lambda text: len(text))

	# prints length statistic detail
	print(data_set.length.describe(), '\n')
	print(data_set.groupby('label').length.describe(), '\n')

	# plot messages length
	plot = data_set.length.plot(bins=150, kind='hist')
	plot.get_figure().show()

	# plot message length in each category
	plots = data_set.hist(column='length', by='label', bins=50)  # bins='auto' , makes plot in more detail
	plots[0].get_figure().show()

	nb_detector = cPickle.load(open(FILEPATH + '/models/nb' + MODELFILEPOSTFIX + '.pkl', mode='rb'))
	showModelDetail(nb_detector, data_set['message'], data_set['label'], 'Naive Bayes Classifier')

	svm_detector = cPickle.load(open(FILEPATH + '/models/svm' + MODELFILEPOSTFIX + '.pkl', mode='rb'))
	showModelDetail(svm_detector, data_set['message'], data_set['label'], 'Support Vector Machine Classifier')

	knn_detector = cPickle.load(open(FILEPATH + '/models/knn' + MODELFILEPOSTFIX + '.pkl', mode='rb'))
	showModelDetail(knn_detector, data_set['message'], data_set['label'], 'K Nearest Neighbor Classifier')

	data_set['nb'] = data_set['message'].map(lambda text: nb_detector.predict([text])[0])
	data_set['svm'] = data_set['message'].map(lambda text: svm_detector.predict([text])[0])
	data_set['knn'] = data_set['message'].map(lambda text: knn_detector.predict([text])[0])

	print("\n\t----------------\t Naive Bayes fails on this messages \t----------------\n")
	nb_fp = data_set[['label', 'message']][data_set.nb != data_set.label]
	print(nb_fp)

	print("\n\t----------------\t SVM fails on this messages \t----------------\n")
	svm_fp = data_set[['label', 'message']][data_set.svm != data_set.label]
	print(svm_fp)

	print("\n\t----------------\t K-NN fails on this messages \t----------------\n")
	knn_fp = data_set[['label', 'message']][data_set.knn != data_set.label]
	print(knn_fp)

	print("\n\t----------------\t NB & SVM \t----------------\n")
	fp = data_set[['label', 'message']][(data_set.nb != data_set.label) & (data_set.svm != data_set.label)]
	print(fp)

	print("\n\t----------------\t NB & KNN \t----------------\n")
	fp = data_set[['label', 'message']][(data_set.nb != data_set.label) & (data_set.knn != data_set.label)]
	print(fp)

	print("\n\t----------------\t SVM & KNN \t----------------\n")
	fp = data_set[['label', 'message']][(data_set.svm != data_set.label) & (data_set.knn != data_set.label)]
	print(fp)

	print("\n\t----------------\t Everyone fails on this messages \t----------------\n")
	fp = data_set[['label', 'message']][(data_set.nb != data_set.label) & (data_set.svm != data_set.label) & (data_set.knn != data_set.label)]
	print(fp)


# ************************* #
#                           #
#         Training          #
#                           #
# ************************* #

def train_K_nearest_neighbor(messages):
	'''
	Trains and optimize K Nearest Neighbor Classifier
	:param messages:
	:return: None
	'''
	# pipeline parameters to automatically explore and tune
	params = {
		'tfidf__use_idf': (True, False),
		'bow__analyzer': (splitToLemmas_NoStopWord, splitToLemmas, splitToTokens, 'word'),
		'classifier__n_neighbors': numpy.arange(start=1, stop=100),
		'classifier__weights': ['uniform', 'distance']
	}

	# run optimization and cross validation
	knn_detector = optimizeParameters(
		data_set=messages,
		classifier=KNeighborsClassifier(),  # train on TF-IDF vectors with K Nearest Neighbor Classifier
		parameters=params,
		title='K Nearest Neighbor Classifier'
	)

	# save model to file, so we can just use it later
	saveModelToFile(knn_detector, 'knn' + MODELFILEPOSTFIX + '.pkl')


def train_multinomial_nb(messages):
	'''
	Trains and optimize Naive Bayes Classifier
	:param messages:
	:return: None
	'''
	# pipeline parameters to automatically explore and tune
	params = {
		'tfidf__use_idf': (True, False),
		'bow__analyzer': (splitToLemmas_NoStopWord, splitToLemmas, splitToTokens, 'word'),
	}

	# run optimization and cross validation
	nb_detector = optimizeParameters(
		data_set=messages,
		classifier=MultinomialNB(),  # train on TF-IDF vectors with Naive Bayes Classifier
		parameters=params,
		title='Naive Bayes Classifier'
	)

	# save model to file, so we can just use it
	saveModelToFile(nb_detector, 'nb' + MODELFILEPOSTFIX + '.pkl')


def train_svm(messages):
	'''
	Trains and optimize Support Vector Machine Classifier
	:param messages:
	:return: None
	'''
	# pipeline parameters to automatically explore and tune
	params = {
		'tfidf__use_idf': (True, False),
		'bow__analyzer': (splitToLemmas_NoStopWord, splitToLemmas, splitToTokens, 'word', 'char'),
		'classifier__C': [1, 10, 100, 1000],
		'classifier__gamma': ['auto', 0.001, 0.0001],
		'classifier__kernel': ['rbf', 'linear'],
	}

	# run optimization and cross validation
	svm_detector = optimizeParameters(
		data_set=messages,
		classifier=SVC(),  # train on TF-IDF vectors with Support Vector Machine Classifier
		parameters=params,
		title='Support Vector Machine Classifier'
	)

	# save model to file, so we can just use it
	saveModelToFile(svm_detector, 'svm' + MODELFILEPOSTFIX + '.pkl')


def main(args):
	# check if models exist, if not run training
	if (os.path.isfile(FILEPATH + '/models/nb' + MODELFILEPOSTFIX + '.pkl') == False):
		print("Creating Naive Bayes Model .....")
		train_multinomial_nb(DATASET)

	if (os.path.isfile(FILEPATH + '/models/svm' + MODELFILEPOSTFIX + '.pkl') == False):
		print("Creating Support Vector Machine Model .....")
		train_svm(DATASET)

	if (os.path.isfile(FILEPATH + '/models/knn' + MODELFILEPOSTFIX + '.pkl') == False):
		print("Creating K Nearest Neighbor Model .....")
		train_K_nearest_neighbor(DATASET)

	if args.detail:
		showDeatils(DATASET)

	if args.message != "" and args.message is not None:
		print("\n\n\t============ Final Result ============\n")
		print("Message:\n")
		print(args.message)
		print("\n\t------------ ------------ ------------\n")
		prediction = predict(args.message)
		print("Support Vector Machine:\t", prediction[0])
		print("Naive Bayes:\t\t", prediction[1])
		print("K Nearest Neighbor:\t", prediction[2])


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"-m",
		"--message",
		type=str,
		help="The message to be checked for Spam.",
	)
	parser.add_argument(
		"-d",
		"--detail",
		action="store_true",
		help="Show models detail."
	)
	parser.add_argument(
		"-p",
		"--postfix",
		type=str,
		default=MODELFILEPOSTFIX,
		help='Model file postfix, what will be after "algorithm name" and before ".pkl". please note the under lines.(example: -p "_optimized-model-char")'
	)
	args = parser.parse_args()

	if args.postfix is not None and args.postfix != "" and args.postfix != MODELFILEPOSTFIX:
		MODELFILEPOSTFIX = args.postfix

	main(args)
