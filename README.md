# Credit Card Fraud Detection Using Exploratory Data Analysis, Data Visualization and Deep Learning

## 1. Project Overview

Credit card fraud is a major challenge in the financial industry, where fraudulent transactions can cause significant financial losses. Identifying fraudulent transactions accurately is essential for improving transaction security and reducing financial risk.

This project uses **Exploratory Data Analysis (EDA), statistical analysis, data visualization, and a Multilayer Perceptron (MLP) deep learning model** to analyze credit card transactions and classify them as legitimate or fraudulent.

The project also explores class imbalance, examines feature distributions, preprocesses the data, and evaluates the model using multiple classification metrics and visualizations.

## 2. Objectives

- Perform exploratory data analysis on credit card transaction data.
- Identify missing values, duplicate records, outliers, and class imbalance.
- Visualize feature distributions and relationships between variables.
- Preprocess the dataset using feature scaling and stratified data splitting.
- Build and train a deep learning model for fraud classification.
- Compare selected learning-rate configurations.
- Evaluate the model using classification metrics and performance visualizations.

## 3. Dataset Description

**Dataset:** Credit Card Fraud Detection

**Source:** Kaggle — Machine Learning Group, Université Libre de Bruxelles (ULB)

**Dataset link:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The dataset contains **284,807 transactions and 31 columns**, including the target variable.

### Features

- **Time:** Seconds elapsed between each transaction and the first transaction in the dataset.
- **V1–V28:** Numerical features obtained through Principal Component Analysis (PCA).
- **Amount:** Transaction amount.
- **Class:** Target variable indicating whether a transaction is legitimate or fraudulent.

### Target variable

| Class | Meaning |
|---|---|
| 0 | Legitimate transaction |
| 1 | Fraudulent transaction |

The dataset is highly imbalanced, with legitimate transactions substantially outnumbering fraudulent transactions.

**Note:** The dataset is not included in this repository. Download it from Kaggle and upload `creditcard.csv` to the notebook environment before running the analysis.

## 4. Technologies Used

- Python
- Google Colab / Jupyter Notebook
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- TensorFlow / Keras

## 5. Project Workflow

### Step 1: Data Loading

Load the dataset into a Pandas DataFrame and inspect its dimensions, columns, and data types.

### Step 2: Exploratory Data Analysis

- Examine the dataset structure and descriptive statistics.
- Check missing values and duplicate records.
- Analyze the distribution of legitimate and fraudulent transactions.
- Investigate transaction amounts and potential outliers.
- Explore correlations and selected feature distributions.

### Step 3: Data Visualization

The project includes visualizations such as:

- Class distribution of legitimate and fraudulent transactions.
- Distribution of transaction amounts.
- Correlation heatmap.
- Feature distributions for selected PCA-derived variables.
- Transaction amount versus time.
- Transaction amount boxplots, including comparisons by class.

Each visualization is used to understand patterns, distributions, relationships, or potential anomalies in the dataset.

### Step 4: Data Preprocessing

- Separate the input features and target variable.
- Split the dataset into training, validation, and test sets using stratified sampling.
- Standardize features using `StandardScaler`.
- Apply class weights calculated from the training data to help address class imbalance.

The approximate split is 70% training, 15% validation, and 15% testing.

### Step 5: Deep Learning Model

A Multilayer Perceptron (MLP) neural network is implemented using TensorFlow/Keras.

The architecture consists of:

- Input layer with 30 features.
- Dense layer with 64 neurons and ReLU activation.
- Dropout layer with a rate of 0.30.
- Dense layer with 32 neurons and ReLU activation.
- Dropout layer with a rate of 0.20.
- Dense layer with 16 neurons and ReLU activation.
- Output layer with one neuron and sigmoid activation.

**Training configuration:**

- Optimizer: Adam
- Loss function: Binary cross-entropy
- Batch size: 2048
- Maximum epochs: 30
- Early stopping based on validation PR-AUC
- Learning-rate reduction when validation loss stops improving

Selected learning rates are compared as part of basic hyperparameter tuning.

### Step 6: Model Evaluation

The trained model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Average precision (PR-AUC)

The notebook also generates the following performance visualizations:

- Training and validation loss versus epochs.
- Training and validation accuracy versus epochs.
- Confusion matrix.
- Receiver Operating Characteristic (ROC) curve.
- Precision-Recall curve.

These results help assess the model's ability to identify fraudulent transactions, particularly in an imbalanced dataset.

## 6. Results and Discussion

The notebook reports the model's evaluation metrics and displays its performance visualizations. The results can be used to examine the balance between detecting fraudulent transactions and correctly identifying legitimate transactions.

**Final model metrics:** Refer to the evaluation output in the notebook and the corresponding results section in the project report.

## 7. How to Run the Project

1. Clone or download this repository.
2. Open `Credit_Card_Fraud_Detection.ipynb` in Google Colab or Jupyter Notebook.
3. Download `creditcard.csv` from the Kaggle dataset link provided above.
4. Upload the CSV file when prompted by the notebook.
5. Run the notebook cells in order, from data loading through model evaluation.
6. Review the generated EDA plots, training curves, and evaluation metrics.

### Required libraries

Install the required libraries if they are not already available:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

## 8. Repository Structure

```text
credit-card-fraud-detection/
│
├── README.md
├── Credit_Card_Fraud_Detection.ipynb
├── Credit_Card_Fraud_Detection_Report.pdf
│
└── data/
    └── README.md
```

The `data/README.md` file can contain the dataset source and download instructions. If you export your visualizations separately, you can also add a `results/` folder.

## 9. Conclusion

This project demonstrates an end-to-end workflow for credit card fraud detection, from exploratory data analysis and statistical visualization to preprocessing, deep learning model development, and evaluation.

The MLP model provides a framework for learning patterns in transaction features and classifying transactions. Evaluation metrics such as recall, F1-score, ROC-AUC, and average precision help assess its performance beyond accuracy alone.

Future improvements may include experimenting with alternative neural network architectures, resampling methods, threshold optimization, and additional model-comparison techniques.

## 10. References

1. Machine Learning Group, Université Libre de Bruxelles. *Credit Card Fraud Detection Dataset*. Kaggle. https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. TensorFlow. *TensorFlow Documentation*. https://www.tensorflow.org/
3. Scikit-learn. *Scikit-learn Documentation*. https://scikit-learn.org/stable/
4. Pandas. *Pandas Documentation*. https://pandas.pydata.org/docs/
5. Matplotlib. *Matplotlib Documentation*. https://matplotlib.org/stable/

---

**Project Type:** Academic Case Study  
**Domain:** Machine Learning and Deep Learning  
**Programming Language:** Python  
**Notebook Environment:** Google Colab / Jupyter Notebook
