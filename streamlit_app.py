import streamlit as st
import joblib
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Load the trained model and vectorizer
model = joblib.load(r'model.pkl')
countvec = joblib.load(r'countvec.pkl')

# Initialize preprocessing tools
lemmatizer = WordNetLemmatizer()
en_stopwords = stopwords.words('english')

def preprocess_text(text):
    # Remove prefixes like "Source - "
    text = re.sub(r"^[^-]*-\s", "", text)
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r"([^\w\s])", "", text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in en_stopwords]
    # Join back to string for vectorization
    return ','.join(tokens)

# Streamlit app interface
st.title("📰 Fake News Classifier")
st.markdown("Enter a news article text below to classify it as **Fake News** or **Factual News**.")

# Text input
user_input = st.text_area("News Article Text", height=200, placeholder="Paste your news text here...")

# Classify button
if st.button("🔍 Classify News"):
    if user_input.strip():
        # Preprocess the input
        processed_text = preprocess_text(user_input)
        
        # Vectorize
        vectorized = countvec.transform([processed_text])
        
        # Predict
        prediction = model.predict(vectorized)[0]
        
        # Display result
        if prediction == 'Fake News':
            st.error(f"🚨 Prediction: **{prediction}**")
            st.write("This article appears to be fake news. Always verify information from reliable sources.")
        else:
            st.success(f"✅ Prediction: **{prediction}**")
            st.write("This article appears to be factual news.")
        
        # Optional: Show processed text
        with st.expander("See Processed Text"):
            st.write(processed_text)
    else:
        st.warning("Please enter some text to classify.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | Model trained on bag-of-words features with Logistic Regression")