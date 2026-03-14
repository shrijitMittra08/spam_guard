import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data.csv')

df_training, df_testing = train_test_split(df, test_size = 0.2, random_state = 42)

print('Training data:')
print(df_training.columns)
print()
print('Testing data:')
print(df_testing.columns)

print(df_training['label'].value_counts())
print(df_testing['label'].value_counts())

df_training.to_csv('training_data.csv', index = False)
df_testing.to_csv('testing_data.csv', index = False)

