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
from nltk import TreebankWordTokenizer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import urllib
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

# Plotting libraries
import plotly.express as px
import plotly.graph_objects as go

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
train_data = pd.read_csv("resources/train2.csv")
test_data = pd.read_csv("resources/test_with_no_labesls.csv")

@st.cache
def get_data(filename):
	twitter_data = pd.read_csv(filename)

	return twitter_data

#Data pre-processing functions


# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Climate Change Belief Tweet Classifier")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Summary", "EDA", "Prediction", "Information"]
	selection = st.sidebar.selectbox("Choose option", options)

	if selection == "Summary":
		st.subheader("Team Members")
		st.markdown(" * **Pabatso Tejane** ")
		st.markdown(" * **Olefile Ramoitheki** ")
		st.markdown(" * **Euphrasia Mampuru** ")
		st.markdown(" * **Nqobile Ncube** ")
		st.markdown(" * **Collen Bothma** ")

		st.subheader("Project Problem Statement")
		st.info("The client has tasked the team with creating a tweet classifier: that will assist in identifying potential customers for their eco friendly products and services")

	if selection == "EDA":
		st.subheader("Exploratory data analysis")
		st.markdown("The graph below shows the distribution of the four possible sentiments which are represented in the raw data.")
			
	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw_data[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")

		st.subheader("Prediction model selection")

	#if st.checkbox('Model 1'): 
	#if st.checkbox('Model 2'): 
	#if st.checkbox('Model 3'): 
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
