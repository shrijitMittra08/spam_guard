# 🛡️ Spam_Guard

Spam_Guard is an advanced, end-to-end Machine Learning pipeline designed to classify emails as **Ham (Safe)** or **Spam** with high accuracy. The project leverages Natural Language Processing (NLP) techniques, particularly TF-IDF vectorization, combined with powerful classification algorithms like Support Vector Machines (SVM) and Logistic Regression.

### 📊 Datasets
The training and testing data for this project (including the famous **Enron Email Corpus**) can be accessed here:
👉 [Datasets GDrive Link](https://drive.google.com/drive/folders/1xmLkKAGCCBAjuuBljmb_mwaD-Y9xZi1L?usp=drive_link)

---

## 🚀 Project Pipeline

### 1. Exploratory Data Analysis (EDA)
Before training, the dataset underwent extensive EDA to understand the distribution of words, characters, and sentences between Ham and Spam emails.
* **Ham Insights:** Dominated by conversational Enron-specific terminology (`enron`, `ect`, `hou`, `vince`). 
* **Spam Insights:** Heavily populated with pharmaceutical, financial, and tech-scam terms (`pill`, `nummg`, `currency`, `software`).
* **Feature Engineering:** Calculated `num_chars`, `num_words`, and `num_sentences` to analyze structural differences between classes.

### 2. Advanced Text Preprocessing
Raw email text is incredibly messy. We built a custom, highly robust `clean_text` Python function that normalizes the data:
* **Regex Filtering:** Replaces loose URLs with `webaddr`, emails with `emailaddr`, numbers with `num`, and monetary symbols with `currency`.
* **Punctuation Stripping:** Completely removes all special characters (`!@#$%^&*`).
* **Stopwords & Lemmatization:** Uses `nltk` to remove useless conversational filler (and custom stopwords like `http`, `www`, `com`) and lemmatizes words down to their root dictionary form.
* **Safety Checks:** Automatically drops any completely blank strings or `NaN` rows that result from cleaning.

### 3. TF-IDF Vectorization
Instead of using Bag of Words (BoW) or Word2Vec, this project uses **Term Frequency-Inverse Document Frequency (TF-IDF)** to convert text into mathematical sparse matrices. 
* **Parameters:** `max_features=5000` (focusing only on the top 5,000 most statistically significant words) and `min_df=5`.
* **Prevention of Data Leakage:** The vectorizer is `fit` *only* on the training data, and strictly uses `.transform()` on the testing data to simulate a real-world testing environment.

### 4. Model Training & Evaluation
The TF-IDF sparse matrices and labels are saved as `.pkl` files via `joblib` for rapid loading. They are then fed into `scikit-learn` classification models:
* **Logistic Regression** (`max_iter=1000`)
* **Linear Support Vector Classifier (LinearSVC)**

*(Accuracy scores and Confusion Matrices are generated to evaluate the final performance.)*

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
├── *.csv                       # Remaining processed/split CSV files
├── *.pkl                       # Serialized models, matrices, and vectorizer
└── *.py                        # Python scripts for preprocessing and training
```