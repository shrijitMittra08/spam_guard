import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt', quiet=True)
stop_words = set(stopwords.words('english'))

custom_stopwords = ['http', 'https', 'www', 'com', 'org', 'net', 'co']
stop_words.update(custom_stopwords)
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    
    text = str(text).lower()

    url_regex = r'(https?:\/\/\S+|www\.\S+)'
    email_regex = r'[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}'
    num_regex = r'\d+'
    currency_regex = r'[$£€]'

    text = re.sub(url_regex, 'webaddr', text)
    text = re.sub(email_regex, 'emailaddr', text)
    text = re.sub(num_regex, 'num', text)
    text = re.sub(currency_regex, 'currency ', text)

    text = text.replace('escapenumber', 'num')

    text = re.sub(r'[^a-z\s]', '', text)

    words = text.split()

    # len(i) > 1 to filter out leftover tags such as x, e, etc. and seen in EDA
    clean_words = [lemmatizer.lemmatize(i) for i in words if i not in stop_words and len(i) > 1]

    return ' '.join(clean_words)

print('Initializing SpamGuard')
print()

print('Loading models')

try:
    ensemble = joblib.load('models/ensemble_model.pkl')
    vectorizer = joblib.load('vectors/tfidf_vectorizer.pkl')
    sel = joblib.load('models/sel.pkl')
    print('Models loaded successfully.')
    while True:
        try:
            text = input("Enter email (exit to quit): ")
            if text.lower().strip() == 'exit':
                break
            text = text.strip()
            if not text:
                print("Enter an email!")
                continue
            
            cleaned_text = clean_text(text)
            vec_text = vectorizer.transform([cleaned_text])
            features = sel.transform(vec_text)

            pred = ensemble.predict(features)[0]
            prob = ensemble.predict_proba(features)[0]

            print('Analysis Result:')

            if pred == 1:
                print('Verdict: Spam')
                print(f'Confidence: {prob[1]*100:.2f}')
            else:
                print('Verdict: Ham')
                print(f'Confidence: {prob[0]*100:.2f}')

        except Exception as e:
            print(f'Error: {e}')
            
except Exception as e:
    print(f'Error: {e}')
