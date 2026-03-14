import pandas as pd

df1 = pd.read_csv('datasets/email.csv')
df2 = pd.read_csv('datasets/emails.csv')
df3 = pd.read_csv('datasets/combined_data.csv')
df4 = pd.read_csv('datasets/df.csv')
df5 = pd.read_csv('datasets/email_dataset_100k.csv')

print("df1 details:")
print(df1.shape)
print(df1.info())
print(df1.columns)
print()
print("df2 details:")
print(df2.shape)
print(df2.info())
print(df2.columns)
print()
print("df3 details:")
print(df3.shape)
print(df3.info())
print(df3.columns)
print()
print("df4 details:")
print(df4.shape)
print(df4.info())
print(df4.columns)
print()
print("df5 details:")
print(df5.shape)
print(df5.info())
print(df5.columns)

df1 = df1.rename(columns = {'Category': 'label', 'Message': 'text'})
df2 = df2.rename(columns = {'spam': 'label'})
for i in range(2, 110):
    del df2[f'Unnamed: {i}']
df5 = df5.rename(columns = {'raw_text': 'text'})

print("df1 columns:")
print(df1.columns)
print()
print("df2 columns:")
print(df2.columns)
print()
print("df3 columns:")
print(df3.columns)
print()
print("df4 columns:")
print(df4.columns)
print()
print('df5 columns:')
print(df5.columns)

df1 = df1[['text', 'label']]
df3 = df3[['text', 'label']]
df4 = df4[['text', 'label']]
df5 = df5[['text', 'label']]

print("df1 columns:")
print(df1.columns)
print()
print("df2 columns:")
print(df2.columns)
print()
print("df3 columns:")
print(df3.columns)
print()
print("df4 columns:")
print(df4.columns)

combined_df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)

print(combined_df.shape)
print(combined_df.info())
print(combined_df.columns)

combined_df.to_csv("data.csv", index = False)

