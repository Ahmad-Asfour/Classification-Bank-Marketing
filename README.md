# Bank Marketing Campaign Analysis and Prediction

This project aims to analyze and predict whether a bank client will subscribe to a term deposit based on their demographic information, financial status, and previous interactions with the bank’s marketing campaigns. By identifying patterns in the data, this project helps optimize marketing strategies and improve campaign success rates.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Objective](#objective)
4. [Key Challenges](#key-challenges)
5. [Steps and Implementation](#steps-and-implementation)
6. [Results](#results)
7. [Files and Directories](#files-and-directories)
8. [How to Run](#how-to-run)
9. [Technologies Used](#technologies-used)

---

## Project Overview

This project involves:

- **Data Analysis**: Understanding the dataset and exploring relationships between variables.
- **Data Preprocessing**: Cleaning the data, handling missing values, encoding categorical variables, and scaling numeric features.
- **Class Imbalance Handling**: Addressing the imbalance in the target variable using SMOTE (Synthetic Minority Oversampling Technique).
- **Model Training**: Using Logistic Regression and Random Forest classifiers to classify clients as likely or unlikely to subscribe.
- **Evaluation**: Measuring the models' performance through metrics such as accuracy, precision, recall, and F1-score.

---

## Dataset

- **Source**: [Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- **Description**: The dataset contains information about clients contacted as part of a bank marketing campaign. It includes 16 features, such as demographic information, financial status, and specifics of the latest contact.

---

## Objective

To develop a predictive model that classifies bank clients as likely or unlikely to subscribe to a term deposit based on their demographic, financial, and interaction data.

---

## Key Challenges

- **Class Imbalance**: The dataset has significantly more non-subscribers (`"no"`) than subscribers (`"yes"`).
- **Feature Selection**: Identifying the most relevant features for prediction.
- **Overfitting**: Ensuring the model generalizes well on unseen data.

---

## Steps and Implementation

### 1. Load the Data and Initial Inspection
- Load the dataset and inspect its structure using `df.info()`.
- Analyze and visualize relationships between features and the target variable (`y`).

### 2. Data Cleaning and Preprocessing
- **Encoding Categorical Variables**: Use one-hot encoding for variables like `job`, `marital`, and `contact`.
- **Binary Mapping**: Convert binary variables like `housing` and `loan` to 0/1.
- **Scaling Numeric Features**: Standardize features like `age`, `balance`, and `duration` using `StandardScaler`.

### 3. Handle Class Imbalance
- Apply **SMOTE** to oversample the minority class (`"yes"`) in the training dataset.

### 4. Model Training and Evaluation

#### Logistic Regression
- Trained a **Logistic Regression** model with and without class balancing.
- Achieved a recall of 81% for the minority class after applying SMOTE, indicating improved sensitivity to subscribers.

#### Random Forest
- Trained a **Random Forest** classifier, testing it both with and without class balancing.
- Evaluated using metrics such as accuracy, precision, recall, and F1-score.
- Results showed high overall accuracy but lower recall for the minority class compared to Logistic Regression.

### 5. Save Processed Data
- Saved the cleaned and split data as `.csv` and `.npy` files for easy reuse.

---

## Results

### Logistic Regression (After SMOTE)
| Metric               | Value              |
|----------------------|--------------------|
| **Accuracy**         | 83%               |
| **Precision (Class 1)** | 38%             |
| **Recall (Class 1)**    | 81%             |
| **F1-Score (Class 1)**  | 52%             |

### Random Forest (After SMOTE)
| Metric               | Value              |
|----------------------|--------------------|
| **Accuracy**         | 90%               |
| **Precision (Class 1)** | 58%             |
| **Recall (Class 1)**    | 39%             |
| **F1-Score (Class 1)**  | 47%             |

### Observations
- **Logistic Regression**: Higher recall for the minority class makes it better for identifying subscribers, though precision is lower.
- **Random Forest**: Higher precision and accuracy overall but struggles with recall for the minority class.

---

## Files and Directories

- `bank.csv`: Original dataset.
- `processed_data.zip`: Preprocessed `.csv` files for train-test split.
- `processed_data_npy.zip`: Preprocessed `.npy` files for train-test split.
- `model.ipynb`: Notebook containing Logistic Regression and Random Forest implementations.

---

## How to Run

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-repository/bank-marketing-prediction.git
   cd bank-marketing-prediction
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Data Exploration**
   ```bash
   jupyter notebook BankData_exploration.ipynb
   ```

4. **Run Logistic Regression Model**
   ```bash
   jupyter notebook BankLogisticRegModel.ipynb
   ```

5. **Run Random Forest Model**
   ```bash
   jupyter notebook BankRandomForestModel.ipynb
   ```

---

## Technologies Used

- **Python**: Core programming language.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib/Seaborn**: For data visualization.
- **Scikit-learn**: For preprocessing, training, and evaluation.
- **Imbalanced-learn**: For handling class imbalance (SMOTE).
- **Jupyter Notebook**: For interactive development.

