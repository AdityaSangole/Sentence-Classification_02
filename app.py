import streamlit as st
import pickle

# Load the trained model, vectorizer, and accuracy score
with open('knn_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

with open('accuracy_score.pkl', 'rb') as acc_file:
    accuracy = pickle.load(acc_file)

# Load classification report
with open('classification_report.txt', 'r') as file:
    classification_report_text = file.read()

# Function to preprocess text
def pre_process(text):
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    import string

    nltk.download('punkt')
    nltk.download('stopwords')

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word not in string.punctuation]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    return ' '.join(stemmed_tokens)

# Streamlit UI
st.title("Active vs. Passive Voice Detection")

# Display model accuracy score
st.write(f"Model Accuracy: {accuracy:.2f}")

# Display classification report
st.write("### Classification Report")
st.text(classification_report_text)

# Provide brief information about the classification process
st.write("""
### Classification Process

1. **Preprocessing**: The text data is first tokenized into words. Stop words (common words that do not contribute much meaning) are removed, and stemming is applied to reduce words to their root forms.

2. **Vectorization**: The preprocessed text is converted into numerical vectors using TF-IDF (Term Frequency-Inverse Document Frequency). This helps in quantifying the importance of words in the sentences.

3. **Model Training**: A machine learning model (K-Nearest Neighbors in this case) is trained on the vectorized text data to distinguish between active and passive voice sentences.

4. **Prediction**: The trained model is used to classify new sentences based on their vectorized representation.

### Applications

- **Education**: Helps in automated grading and feedback for writing assignments.
- **Content Creation**: Assists in adjusting writing style according to desired tone.
- **Text Analysis**: Useful in linguistic research and improving natural language processing tools.
""")

# Input and classification
st.write("Enter a sentence to check if it's in the active or passive voice:")

input_sentence = st.text_input("Sentence")

if st.button("Classify"):
    if input_sentence:
        processed_text = pre_process(input_sentence)
        text_vector = vectorizer.transform([processed_text])
        prediction = model.predict(text_vector)
        result = "Active Voice" if prediction == 0 else "Passive Voice"
        st.write(f"The sentence is classified as: {result}")
    else:
        st.write("Please enter a sentence to classify.")
