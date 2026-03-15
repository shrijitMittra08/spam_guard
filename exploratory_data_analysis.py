import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

df = pd.read_csv('preprocessed_training_data.csv')

print(df['label'].value_counts())
print()
print(df['label'].isnull().sum())
print(df['clean_text'].isnull().sum())
print()
print(df.columns.to_list())
print()
print("text column datatype:")
print(df['text'].dtype)
print("label column datatype:")
print(df['label'].dtype)

plt.figure()
plt.pie(df['label'].value_counts(), labels = ['ham', 'spam'], autopct = '%0.2f')
plt.show()

print("Statistics for ham emails:")
print()
print(df[df['label'] == 0][['num_chars', 'num_words', 'num_sentences']].describe())
print()
print("Statistics for spam emails:")
print()
print(df[df['label'] == 1][['num_chars', 'num_words', 'num_sentences']].describe())

plt.figure(figsize=(12, 6))
sns.histplot(
    data=df, 
    x='num_chars', 
    hue='label', 
    log_scale=True, 
    bins=50, 
    kde=True,
    palette={0: 'green', 1: 'red'},
    alpha=0.5
)
plt.title('Distribution of Email Lengths')
plt.xlabel('Number of Characters (Log 10)')
plt.ylabel('No. of emails')
plt.legend(["Spam", "Ham"])
plt.show()

plt.figure()
sns.heatmap(df[['label', 'num_chars', 'num_words', 'num_sentences']].corr(method = 'spearman'), annot=True)
plt.title('Feature Correlation')
plt.show()

custom_stopwords = ['num', 'webaddr', 'emailaddr', 'currency', 'escapelong', 'numd', 'nummg', 'aug']
for i in range(97, 123):
    custom_stopwords.append(f'{chr(i)+'num'}')

spam_wc = WordCloud(
     width=800, 
     height=400, 
     background_color='black', 
     colormap='Reds',
     stopwords=custom_stopwords,
     max_words=100
).generate(df[df['label'] == 1]['clean_text'].str.cat(sep = ' '))

plt.figure()
plt.imshow(spam_wc, interpolation='bilinear')
plt.title('Spam Emails Word Cloud')
plt.axis('off')
plt.show()

null_count = df['clean_text'].isna().sum()
empty_string_count = (df['clean_text'].fillna('').str.strip() == '').sum()

print(f"Number of NaN / Null rows: {null_count}")
print(f"Number of completely blank strings: {empty_string_count}")

