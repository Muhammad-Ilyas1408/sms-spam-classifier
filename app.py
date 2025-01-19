# Importing necessary libraries
import streamlit as st                  # Streamlit library for building the web app
import pickle                           # Library to load saved model and vectorizer
import string                           # For handling punctuation in text preprocessing
from nltk.corpus import stopwords       # To remove stopwords during text preprocessing
import nltk                                    # NLTK library for tokenization
from nltk.stem.porter import PorterStemmer     # To apply stemming on text

# Download stopwords, and unkt_tabcls
nltk.download('stopwords')
nltk.download('punkt_tab')

# Create an object for PorterStemmer
ps = PorterStemmer()

# Define a function to preprocess text data
def transform_text(text):
    # Convert the entire text to lowercase to ensure uniformity
    text = text.lower()

    # Tokenize the text into individual words
    text = nltk.word_tokenize(text)

    # Initialize an empty list to store alphanumeric words
    y = []

    # Iterate through each token and append alphanumeric tokens to the list
    for i in text:
        if i.isalnum():
            y.append(i)

    # Update 'text' to contain only alphanumeric tokens
    text = y[:]
    y.clear()

    # Remove stopwords and punctuation from the text
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    # Update 'text' to contain tokens without stopwords and punctuation
    text = y[:]
    y.clear()

    # Apply stemming to each token in the text
    for i in text:
        y.append(ps.stem(i))

    # Join the stemmed tokens back into a single string
    text = " ".join(y)

    # Return the preprocessed text
    return text

# Load the pre-trained TF-IDF vectorizer from file
tfidf = pickle.load(open('vectorizer.pkl','rb'))

# Load the pre-trained model from file
model = pickle.load(open('model.pkl','rb'))

# Set up the Streamlit app
st.title("Email/SMS Spam Classifier")

# Input field for users to enter their message
input_sms = st.text_area("Enter the message to classify it as spam or ham:")

# When the predict button is pressed, perform the following steps
if st.button('Predict'):
    # 1. Preprocess the input text using the transform_text function
    transformed_sms = transform_text(input_sms)
    
    # 2. Vectorize the preprocessed text using the TF-IDF vectorizer
    vector_input = tfidf.transform([transformed_sms])
    
    # 3. Predict the result using the loaded model
    result = model.predict(vector_input)[0]
    
    # 4. Display the result (Spam or Not Spam)
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")