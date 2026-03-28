import streamlit as st
import joblib
import re
from pathlib import Path

# Load the trained model and vectorizer
BASE = Path(__file__).resolve().parent
model = joblib.load(BASE / 'model.pkl')
countvec = joblib.load(BASE / 'countvec.pkl')

# Basic English stopwords set to avoid NLTK data requirements
BASIC_STOPWORDS = {
    'a','about','above','after','again','against','all','am','an','and','any','are','as','at','be','because','been','before','being','below','between','both','but','by',
    'could','did','do','does','doing','down','during','each','few','for','from','further','had','has','have','having','he','her','here','hers','herself','him','himself',
    'his','how','i','if','in','into','is','it','its','itself','just','me','more','most','my','myself','no','nor','not','now','of','off','on','once','only','or','other','our',
    'ours','ourselves','out','over','own','same','she','should','so','some','such','than','that','the','their','theirs','them','themselves','then','there','these','they','this',
    'those','through','to','too','under','until','up','very','was','we','were','what','when','where','which','while','who','whom','why','with','you','your','yours','yourself','yourselves'
}

def preprocess_text(text):
    # Remove prefixes like "Source - "
    text = re.sub(r"^[^-]*-\s", "", text)
    text = text.lower()
    text = re.sub(r"([^\w\s])", "", text)
    tokens = text.split()
    tokens = [token for token in tokens if token and token not in BASIC_STOPWORDS]
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