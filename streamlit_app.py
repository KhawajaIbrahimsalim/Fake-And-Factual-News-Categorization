import streamlit as st
import joblib
import re
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Fake News Classifier",
    page_icon="📰",
    layout="wide",
)

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

# Sample news examples for quick testing
SAMPLE_FAKE = (
    "Breaking: Scientists confirm that a secret plant extract cures all diseases overnight. "
    "This government agency does not want you to know the truth, so share this article now before it is removed."
)

SAMPLE_FACTUAL = (
    "The local council announced a new recycling program on Tuesday to reduce landfill waste. "
    "Officials said the initiative will launch next month after community consultation."
)

EXAMPLE_GUIDE = (
    "Fake news often uses dramatic or sensational language, urgent calls to action, and claims without credible sources. "
    "Factual news usually reports an event, cites institutions or experts, and keeps a neutral tone."
)


def preprocess_text(text):
    # Remove prefixes like "Source - "
    text = re.sub(r"^[^-]*-\s", "", text)
    text = text.lower()
    text = re.sub(r"([^\w\s])", "", text)
    tokens = text.split()
    tokens = [token for token in tokens if token and token not in BASIC_STOPWORDS]
    return ','.join(tokens)


def classify_text(text):
    processed = preprocess_text(text)
    vectorized = countvec.transform([processed])
    prediction = model.predict(vectorized)[0]
    probability = None
    if hasattr(model, 'predict_proba'):
        try:
            probability = model.predict_proba(vectorized)[0].max()
        except Exception:
            probability = None
    return prediction, probability, processed


# Main layout
st.title("📰 Fake News Classifier")
st.markdown(
    "Use this app to evaluate whether a news article reads like **Fake News** or **Factual News**. "
    "Paste a paragraph or use one of the sample examples below."
)

with st.container():
    left, right = st.columns([3, 1])
    with left:
        st.subheader("How to use")
        st.markdown(
            "1. Paste a news article or statement into the text box.\n"
            "2. Press **Classify News** to see the prediction.\n"
            "3. Review the result and the guidance below to understand why the model made that decision."
        )
        st.info(
            "Best suited for short news snippets, headlines, or paragraphs that resemble news reporting. "
            "Avoid non-news text, jokes, or quotes without context."
        )

    with right:
        st.subheader("Why this app")
        st.markdown(
            "- Helps identify suspicious news language.\n"
            "- Demonstrates a simple machine learning pipeline.\n"
            "- Provides clear examples for evaluators."
        )
        st.success("Engineered for clarity with a consistent input flow.")

st.markdown("---")

if 'news_text' not in st.session_state:
    st.session_state.news_text = ''

button_cols = st.columns([1, 1, 1])
with button_cols[0]:
    if st.button("Use Fake News Example"):
        st.session_state.news_text = SAMPLE_FAKE
with button_cols[1]:
    if st.button("Use Factual News Example"):
        st.session_state.news_text = SAMPLE_FACTUAL
with button_cols[2]:
    if st.button("Clear Text"):
        st.session_state.news_text = ''

user_input = st.text_area(
    "News Article Text",
    value=st.session_state.news_text,
    height=220,
    placeholder="Paste your news text here...",
    key='news_text',
)

if st.button("🔍 Classify News"):
    if user_input.strip():
        prediction, probability, processed_text = classify_text(user_input)

        if prediction == 'Fake News':
            st.error(f"🚨 Prediction: **{prediction}**")
            st.write("This article appears to be fake news. Always verify information from trusted sources.")
        else:
            st.success(f"✅ Prediction: **{prediction}**")
            st.write("This article appears to be factual news based on the language patterns in the text.")

        if probability is not None:
            st.write(f"Confidence: **{probability * 100:.1f}%**")

        with st.expander("Prediction details"):
            st.write("**Processed words used for classification:**")
            st.write(processed_text)
            st.markdown(
                "**Hint:** Fake news examples often contain sensational phrases, claims of secrecy, or urgency. "
                "Factual news examples generally use measured language and refer to real events or institutions."
            )
    else:
        st.warning("Please enter some text to classify.")

st.markdown("---")

with st.expander("Example guidance"):
    st.write(EXAMPLE_GUIDE)
    st.write(
        "**Example fake article:** urgent claim, dramatic wording, no source references."
        " **Example factual article:** neutral tone, statements about real programs or government announcements."
    )

with st.sidebar:
    st.header("About")
    st.markdown(
        "This classifier is built with a bag-of-words vectorizer and a trained model. "
        "It is designed as an educational demo, not a definitive fact-checking service."
    )
    st.markdown("**Model details**")
    st.write("- Vectorizer: CountVectorizer\n- Model: Logistic Regression")
    st.markdown("---")
    st.markdown("**Evaluation tips**")
    st.write(
        "Enter short news snippets that sound like a headline or article. "
        "Use the sample buttons if you want a quick demonstration."
    )
