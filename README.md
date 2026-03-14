# 🛡️ Spam_Guard

Spam_Guard is an advanced, end-to-end Machine Learning pipeline designed to classify emails as **Ham (Safe)** or **Spam** with high accuracy. The project leverages Natural Language Processing (NLP) techniques, particularly TF-IDF vectorization, combined with powerful classification algorithms like Support Vector Machines (SVM) and Logistic Regression.

### 📊 Datasets
The training and testing data for this project (including the famous **Enron Email Corpus**) can be accessed here:
👉 [Datasets GDrive Link](https://drive.google.com/drive/folders/1xmLkKAGCCBAjuuBljmb_mwaD-Y9xZi1L?usp=drive_link)

---

## 🛠️ Tech Stack
* **Python**
* **Pandas & NumPy** (Data Manipulation)
* **NLTK & Regular Expressions** (Text Processing)
* **Scikit-Learn** (TF-IDF Vectorization, Logistic Regression, LinearSVC, Metrics)
* **Matplotlib & Seaborn** (Data Visualization)
* **Joblib** (Model and Matrix Serialization)

---

## 📁 File Structure

```text
spam_guard/
├── datasets/                   # Located in Google Drive link above
│   ├── email.csv
│   ├── emails.csv
│   ├── combined_data.csv
│   ├── email_dataset_100k.csv
│   └── df.csv
├── notebooks/                  # Jupyter notebooks for EDA and testing
│   ├──dataset_creation.ipynb
│   ├──dataset_split.ipynb
│   ├──data_preprocessing.ipynb
│   ├──exploratory_data_analysis.ipynb
│   ├──data_vectorizer.ipynb                  
├── data.csv
├── training_data.csv
├── testing_data.csv
├── preprocessed_training_data.csv
├── preprocessed_testing_data.csv
├── tfdif_vectorizer.pkl
├── x_training_vector.pkl
├── x_testing_vector.pkl
├── y_training_vector.pkl
├── y_testing_vector.pkl
├── dataset_creation.py
├── dataset_split.py
├── data_preprocessing.py
├── exploratory_data_analysis.py
└── data_vectorizer.py
```
