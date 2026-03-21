# 🛡️ Spam_Guard

Spam_Guard is an advanced, end-to-end Machine Learning pipeline designed to classify emails as **Ham (Safe)** or **Spam** with high accuracy. The project leverages Natural Language Processing (NLP) techniques, particularly TF-IDF vectorization, combined with powerful classification algorithms like Support Vector Machines (SVM) and Logistic Regression.

### 📊 Datasets | Vectors | Models
The training and testing data, vectors and models can be accessed here:
 [Datasets, Vectors & Models GDrive Link](https://drive.google.com/drive/folders/1xmLkKAGCCBAjuuBljmb_mwaD-Y9xZi1L?usp=drive_link)

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
│   ├── dataset_creation.ipynb
│   ├── dataset_split.ipynb
│   ├── data_preprocessing.ipynb
│   ├── exploratory_data_analysis.ipynb
│   ├── data_vectorizer.ipynb
│   ├── ensemble_build.ipynb
│   ├── ensemble_testing.ipynb
│   ├── model_training.ipynb
│   ├── model_testing.ipynb
├── vectors/                    # Located in Google Drive link
│   ├── tfdif_vectorizer.pkl
│   ├── x_training_vector.pkl
│   ├── x_testing_vector.pkl
│   ├── y_training_vector.pkl
│   ├── y_testing_vector.pkl
├── reports/
│   ├── model_report.csv
│   ├── ensemble_report.txt
├── models/
│   ├── ensemble_model.pkl
│   ├── model_KNN.pkl
│   ├── model_Logistic Regression.pkl
│   ├── model_Random Forest.pkl
│   ├── model_Ridge Classifier.pkl
│   ├── model_SVM Poly.pkl
│   ├── model_SVM RBF.pkl
│   ├── sel.pkl  
├── data.csv
├── training_data.csv
├── testing_data.csv
├── preprocessed_training_data.csv
├── preprocessed_testing_data.csv
├── dataset_creation.py
├── dataset_split.py
├── data_preprocessing.py
├── exploratory_data_analysis.py
├── data_vectorizer.py
├── model_training.py
├── model_testing.py
├── ensemble_build.py
├── ensemble_testing.py
└── main.py                       # Actual program for SpamGuard CLI
```
