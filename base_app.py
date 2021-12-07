"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Libraries to be used in data cleaning and model
import nltk
from nltk.tokenize import word_tokenize
from nltk import TreebankWordTokenizer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag
import string
import urllib
from nlppreprocess import NLP
#import en_core_web_sm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

import numpy as np # linear algebra
import matplotlib.pyplot as plt
import re
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Data dependencies
import pandas as pd
import numpy as np

# Plotting libraries
import plotly.express as px
import plotly.graph_objects as go

# Vectorizer
news_vectorizer = open("resources//tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

#@st.cache(allow_output_mutation=True)
def get_data(filename):
	data = pd.read_csv(filename)

	return data

data = pd.read_csv('resources/train.csv')
# Load your raw data
raw_tweets = get_data("train.csv")
tweets = get_data("train.csv")
test_data = get_data("test_with_no_labels.csv")

#Data pre-processing functions

def cleaner(tweet):
    tweet = tweet.lower()
    
    to_del = [
        r"@[\w]*",  # strip account mentions
        r"http(s?):\/\/.*\/\w*",  # strip URLs
        r"#\w*",  # strip hashtags
        r"\d+",  # delete numeric values
        r"rt[\s]+",
        r"U+FFFD",  # remove the "character note present" diamond
    ]
    for key in to_del:
        tweet = re.sub(key, "", tweet)
    
    # strip punctuation and special characters
    tweet = re.sub(r"[,.;':@#?!\&/$]+\ *", " ", tweet)
    # strip excess white-space
    tweet = re.sub(r"\s\s+", " ", tweet)
    
    return tweet.lstrip(" ")

def lemmatizer(df):
    df["length"] = df["message"].str.len()
    df["tokenized"] = df["message"].apply(word_tokenize)
    df["parts-of-speech"] = df["tokenized"].apply(nltk.tag.pos_tag)
    
    def str2wordnet(tag):
        conversion = {"J": wordnet.ADJ, "V": wordnet.VERB, "N": wordnet.NOUN, "R": wordnet.ADV}
        try:
            return conversion[tag[0].upper()]
        except KeyError:
            return wordnet.NOUN
    
    wnl = WordNetLemmatizer()
    df["parts-of-speech"] = df["parts-of-speech"].apply(
        lambda tokens: [(word, str2wordnet(tag)) for word, tag in tokens]
    )
    df["lemmatized"] = df["parts-of-speech"].apply(
        lambda tokens: [wnl.lemmatize(word, tag) for word, tag in tokens]
    )
    df["lemmatized"] = df["lemmatized"].apply(lambda tokens: " ".join(map(str, tokens)))
    
    return df


# Application of the function to clean the tweets
tweets['message'] = tweets['message'].apply(cleaner)
test_data['message'] = test_data['message'].apply(cleaner)

tweets = lemmatizer(tweets)
test_data = lemmatizer(test_data)



# The main function where we will build the actual app
def main():
	#Tweet Classifier App with Streamlit 

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Climate Change Belief Tweet Classifier")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Summary", "EDA", "Prediction", "Information"]
	selection = st.sidebar.selectbox("Choose option", options)

	# Building the "Summary" page
	if selection == "Summary":
		st.subheader("Team Members")
		st.markdown(" * **Pabatso Tejane** ")
		st.markdown(" * **Olefile Ramoitheki** ")
		st.markdown(" * **Euphrasia Mampuru** ")
		st.markdown(" * **Nqobile Ncube** ")
		st.markdown(" * **Collen Bothma** ")

		st.subheader("Project Problem Statement")
		st.info("The client has tasked the team with creating a tweet classifier: that will assist in identifying potential customers for their eco friendly products and services")

	# Building the "EDA" page
	if selection == "EDA":
		st.subheader("Exploratory data analysis")
		st.markdown("The graph below shows the distribution of the four possible sentiments which are represented in the raw data.")
		st.image('images//TwitterValue.PNG')
		st.markdown("The graph below shows the buzzwords used in the tweets.")
		st.image('images//wordcloud.png')
		st.markdown("The table  below shows the counts of the most popular hashtags")
		st.image('images//counts.png')
		st.markdown("The graph below shows the boxplot of the number of words per tweet")
		st.image('images//boxplot.png')


			
	# Building the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw_tweets[['sentiment', 'message']]) # will write the df to the page
		if st.checkbox('Show cleaned data'): # data is hidden if box is unchecked
			st.write(tweets[['sentiment', 'message']]) # will write the df to the page

	if selection == 'Prediction':
		st.info('Classify your tweet here using the ML Models below')
		data_source = ['Select option', 'Data Frame'] ## differentiating between a single text and a dataset inpit
		source_selection = st.selectbox('What to classify?', data_source)

        # Load Our Models
		def load_prediction_models(model_file):
			loaded_models = joblib.load(open(os.path.join(model_file),"rb"))
			return loaded_models

        # Getting the predictions
		def get_keys(val,my_dict):
			for key,value in my_dict.items():
				if val == value:
					return key
			
		
		if source_selection == 'Data Frame':
            ### SINGLE TWEET CLASSIFICATION ###
			st.subheader('DataFrame tweet classification')
			ml_models = ["Linear SVC","Original lr","Multinomial NB","Logistic Regression","K-Neighbours","SGD classifier"]
			model_choice = st.selectbox("Choose ML Model",ml_models)
			text_input = st.file_uploader("Choose a CSV file", type="csv")
			if text_input is not None:
				text_input = pd.read_csv(text_input)
			
			uploaded_dataset = st.checkbox('See uploaded dataset')
			if uploaded_dataset:
				st.dataframe(text_input.head(10))
			
			if st.button('Classify'):
				#st.text("Original test ::\n{}".format(input_text))
				#text_clean = cleaner(input_text) #passing the text through the 'cleaner' function
				#lemma = WordNetLemmatizer()
				#text_lemma = lemma.lemmatize(text_clean)

				text_input['message'] = text_input['message'].apply(cleaner)
				text_input = lemmatizer(tweets)

				X = text_input['message']

				if model_choice == 'Linear SVC':
					predictor = joblib.load(open(os.path.join("LinearSVC.pkl"),"rb"))
					prediction = predictor.predict(X)
                    # st.write(prediction)
				if model_choice == 'Original lr':
					predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
					prediction = predictor.predict(X)
				#elif model_choice == 'Multinomial NB':
					#predictor = load_prediction_models("MultinomialNB2.pkl")
					#prediction = predictor.predict(text_lemma)
					#st.write(prediction)
				if model_choice == 'Logistic Regession':
					predictor = load_prediction_models("LogisticRegression2.pkl")
					prediction = predictor.predict(X)
                    # st.write(prediction)
				if model_choice == 'K-Neighbours':
					predictor = load_prediction_models("KNeighbours2.pkl")
					prediction = predictor.predict(X)
					# st.write(prediction)
				if model_choice == 'SGD Classifier':
					predictor = load_prediction_models("SGDClassifier2.pkl")
					prediction = predictor.predict(X)
					# st.write(prediction)
					#prediction_labels = {'Negative':-1,'Neutral':0,'Positive':1,'News':2}
					#final_result = get_keys(prediction,prediction_labels)
				st.success("Tweet Categorized as: {}".format(prediction))
				#st.success("Text Categorized as: {}".format(prediction))

			#if st.button("Classify"):
			# Transforming user input with vectorizer
			#vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			#predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			#prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			#st.success("Text Categorized as: {}".format(prediction))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
