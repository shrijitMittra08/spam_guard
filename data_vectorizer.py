import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import joblib

df = pd.read_csv('preprocessed_training_data.csv')
df1 = pd.read_csv('preprocessed_testing_data.csv')

x = df['clean_text']
x1 = df1['clean_text']
y = df['label']
y1 = df1['label']

print(x.value_counts().sum(), x1.value_counts().sum())

tfidf = TfidfVectorizer(
    max_features=5000,  
    min_df=10,          
    ngram_range=(1, 1)  # (1,1) -> single words
)

x_vec = tfidf.fit_transform(tqdm(x, desc = 'Training: TF-IDF Vectorization'))
x1_vec = tfidf.transform(tqdm(x1, desc = 'Testing: TF-IDF Vectorization'))

joblib.dump(x_vec, 'x_training_vector.pkl')
joblib.dump(x1_vec, 'x_testing_vector.pkl')
joblib.dump(y, 'y_training_vector.pkl')
joblib.dump(y1, 'y_testing_vector.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

