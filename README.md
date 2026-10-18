# Breast Cancer Prediction using Logistic Regression

A machine learning project that explores breast cancer measurements,
trains a Logistic Regression classifier, and presents predictions
through an interactive Streamlit dashboard.

> **Disclaimer:** This project is for educational and demonstration
> purposes only. It is not a medical diagnostic tool and must not be
> used to make clinical decisions.

## 1. Project Overview

The project uses features computed from digitized images of breast cell
nuclei to classify a case as:

-   **Benign (0):** the non-malignant class
-   **Malignant (1):** the malignant class

The notebook covers data inspection, exploratory data analysis (EDA),
preprocessing, model training, and evaluation. The Streamlit application
lets users adjust feature values and view the model's prediction, output
probabilities, and a live radar chart.

## 2. Objectives

-   Explore and prepare the breast cancer dataset.
-   Understand patterns in selected measurements using visualizations.
-   Train a Logistic Regression classification model.
-   Evaluate predictions using accuracy and a classification report.
-   Save the trained model and supporting preprocessing information.
-   Build an interactive dashboard with Streamlit and Plotly.

## 3. Technology Stack

  -----------------------------------------------------------------------
  Tool / Library                      Purpose
  ----------------------------------- -----------------------------------
  Python                              Main programming language

  Jupyter Notebook                    Data analysis, experimentation, and
                                      model training

  Pandas                              Data loading and manipulation

  NumPy                               Numerical operations

  Matplotlib and Seaborn              Exploratory data visualizations

  Scikit-learn                        Feature scaling, data splitting,
                                      Logistic Regression, and evaluation

  Joblib                              Saving and loading model artifacts

  Streamlit                           Interactive web application

  Plotly                              Interactive radar chart
  -----------------------------------------------------------------------

## 4. Dataset

The project uses `data.csv`, which contains **569 records** and 30
numeric measurement features, alongside an identifier, the diagnosis
label, and an extra empty column.

The target column is `diagnosis`: - `B` is mapped to `0` (Benign). - `M`
is mapped to `1` (Malignant).

The notebook removes the `id` and `Unnamed: 32` columns before training.
Features describe cell-nucleus measurements in three groups: - **Mean:**
average measurement - **SE:** standard error of the measurement -
**Worst:** the dataset's worst/largest measurement values, as
represented by the feature names

## 5. Exploratory Data Analysis

The notebook includes: - Dataset structure, dimensions, data types, and
missing-value checks. - A class-distribution pie chart and count plot. -
A box plot comparing `radius_mean` across diagnosis classes. - A
correlation heatmap for selected mean measurements.

These analyses help inspect the data and explore relationships before
model training.

## 6. Machine Learning Workflow

1.  Load the dataset using Pandas.
2.  Remove the identifier and extra empty column.
3.  Convert the diagnosis labels into numeric classes.
4.  Separate the input features (`X`) from the target (`y`).
5.  Split the data into training and testing sets using an 80:20 ratio
    and `random_state=42`.
6.  Apply `StandardScaler` to scale numeric features.
7.  Train a Logistic Regression classifier.
8.  Generate predictions on the test set.
9.  Evaluate the model using accuracy and a classification report.
10. Save the model, scaler, feature list, medians, and observed minimum
    and maximum values in the `artifacts/` directory.

**Evaluation note:** The current notebook fits `StandardScaler` before
the train/test split. This allows information from the test set to
influence preprocessing. For a more reliable evaluation, split the raw
data first, fit the scaler only on `X_train`, and use that fitted scaler
to transform both `X_train` and `X_test`.

## 7. Streamlit Application

The dashboard provides: - A sidebar containing sliders for the model's
input features. - A live prediction that updates when feature values
change. - Benign and malignant model-output probabilities. - A radar
chart comparing normalized Mean, SE, and Worst measurements for selected
features. - A dark-themed interface and an educational-use disclaimer.

The radar chart normalizes each measurement against its own observed
dataset range. It is intended for visual comparison, not as a measure of
cancer risk.

## 8. Project Structure

After running the model-saving cell in the notebook, the project should
be organized as follows:

``` text
Breast-Cancer-Prediction/
├── app.py
├── breast_cancer.ipynb
├── data.csv
├── requirements.txt
├── README.md
└── artifacts/
    ├── model.pkl
    ├── scaler.pkl
    ├── features.pkl
    ├── medians.pkl
    ├── min_values.pkl
    └── max_values.pkl
```

The `artifacts/` directory is generated by the notebook's model-saving
cell. The Streamlit application expects these files to exist before it
starts.

## 9. Setup and Installation

### Prerequisites

-   Python 3.10 or a compatible Python version
-   Git
-   Jupyter Notebook or JupyterLab

### Step 1: Clone the repository

Replace the URL with your repository URL:

``` bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

Alternatively, download the repository ZIP and extract it.

### Step 2: Create and activate a virtual environment (recommended)

``` bash
python -m venv .venv
```

On Windows:

``` bash
.venv\Scripts\activate
```

On macOS/Linux:

``` bash
source .venv/bin/activate
```

### Step 3: Install dependencies

Create a `requirements.txt` file with the following packages:

``` text
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
plotly
jupyter
```

Then install them:

``` bash
pip install -r requirements.txt
```

For reproducible results, use a tested and pinned dependency list for
your Python environment.

## 10. Run the Project

### Step 1: Train the model and create artifacts

Open `breast_cancer.ipynb` in Jupyter and run its cells in order.
Confirm that the final model-saving cell completes successfully and
creates the `artifacts/` directory with all six files listed above.

### Step 2: Launch the Streamlit dashboard

From the project root directory, run:

``` bash
streamlit run app.py
```

Streamlit will provide a local URL in the terminal, usually
`http://localhost:8501`. Open that address in your browser.

**Important:** Running `app.py` before creating the model artifacts will
result in a model-loading error.

## 11. Model Evaluation

The notebook calculates: - **Accuracy:** the proportion of test examples
classified correctly. - **Classification report:** precision, recall,
F1-score, and support for each class.

The actual evaluation values are printed when the notebook is run. No
fixed performance score is stated here because results should be taken
from the executed notebook and verified with a leakage-free evaluation
setup.

## 12. Limitations and Future Improvements

-   This is an educational prototype, not a clinically validated system.
-   Model performance depends on the dataset, preprocessing, and
    evaluation method.
-   The current notebook should be updated to avoid data leakage during
    scaling.
-   Evaluate with a stratified split and consider cross-validation.
-   Add a confusion matrix and ROC-AUC score for a broader evaluation.
-   Tune model hyperparameters using training data only.
-   Pin dependency versions and document the final tested environment.
-   Add automated tests and a deployment configuration before production
    use.

## 13. Responsible Use

The model's predicted class and output probabilities are machine
learning outputs. They are **not clinically validated estimates of
cancer risk** and should not be interpreted as medical advice. Do not
use this project to diagnose, screen, or make treatment decisions.

## 14. Author

**Yash Tambe**\
Third-Year B.Tech --- Artificial Intelligence & Data Science

------------------------------------------------------------------------

*Built as a machine learning and interactive dashboard project using
Python, Scikit-learn, Streamlit, and Plotly.*
