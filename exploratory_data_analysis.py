import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

df = pd.read_csv('preprocessed_training_data.csv')

#print(df['label'].value_counts())
#df = df.dropna(subset = ['clean_text'])
#print(df['label'].isnull().sum())
#print(df['clean_text'].isnull().sum())
#print(df.columns)
#print(df.columns)
#print()
#print("text column datatype:")
#print(df['text'].dtype)
#print("label column datatype:")
#print(df['label'].dtype)

#plt.figure()
#plt.pie(df['label'].value_counts(), labels = ['ham', 'spam'], autopct = '%0.2f')
#plt.show()

#print(df['label'].isnull().sum())

#print(df.sample(10))
#
#print()
#
#print("Statistics for ham emails:")
#print()
#print(df[df['label'] == 0][['num_chars', 'num_words', 'num_sentences']].describe())
#print()
#print("Statistics for spam emails:")
#print()
#print(df[df['label'] == 1][['num_chars', 'num_words', 'num_sentences']].describe())
#
#print()
#
#plt.figure(figsize = (12, 6))
#sns.histplot(df[df['label'] == 0]['num_chars'])
#sns.histplot(df[df['label'] == 1]['num_chars'], color = 'red')
#plt.show()

#plt.figure()
#sns.pairplot(df, hue = 'text')
#plt.show()

#print(df.columns)

# custom stopwords to prevent frequently occuring words from outweighing other words in the map
#custom_stopwords = ['num', 'webaddr', 'emailaddr', 'currency', 'escapelong']
#
#spam_wc = WordCloud(
#    width=800, 
#    height=400, 
#    background_color='black', 
#    colormap='Reds',
#    stopwords=custom_stopwords,
#    max_words=100
#).generate(df[df['label'] == 1]['clean_text'].str.cat(sep = ' '))
#
#plt.figure(figsize=(10, 5))
#plt.imshow(spam_wc, interpolation='bilinear')
#plt.title('Spam Emails Word Cloud')
#plt.axis('off')
#plt.show()

#plt.figure()
#sns.heatmap(df.corr(numeric_only=True), annot = True)
#plt.show()

#ham_words = []
#for i in df[df['label'] == 0]['clean_text'].tolist():
#    for j in i.split():
#        ham_words.append(j)
#
#print("Total ham emails left:", len(df[df['label'] == 0]))
#print("Total words collected:", len(ham_words))
#
##print(Counter(ham_words))
#
#for word, count in Counter(ham_words).most_common(50):
#    print(f"{word}: {count}")
#
#spam_words = []
#for i in df[df['label'] == 1]['clean_text'].tolist():
#    for j in i.split():
#        spam_words.append(j)
#
#print("Total spam emails left:", len(df[df['label'] == 1]))
#print("Total words collected:", len(spam_words))
#
#for word, count in Counter(spam_words).most_common(50):
#    print(f"{word}: {count}")

# 1. Check for actual Missing Values (NaN)
null_count = df['clean_text'].isna().sum()

# 2. Check for empty strings ("", " ", "   ")
# We use .str.strip() to chop off any invisible spaces before checking if it equals ""
empty_string_count = (df['clean_text'].fillna('').str.strip() == '').sum()

print(f"Number of NaN / Null rows: {null_count}")
print(f"Number of completely blank strings: {empty_string_count}")