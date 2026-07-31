# Heart Disease Classification

An end-to-end machine-learning classification project that demonstrates data understanding, cleaning, exploratory analysis, preprocessing, model comparison, and model persistence for an imbalanced binary-classification problem.

## Project Overview

This project predicts whether a structured patient record belongs to one of two classes:

- **Class 0:** No heart disease
- **Class 1:** Heart disease

The repository was developed as an undergraduate machine-learning portfolio project. It demonstrates a complete workflow from raw data inspection to saving and reloading a reusable scikit-learn pipeline.

This project is intended for education and portfolio demonstration only. It is not intended for clinical diagnosis or medical decision-making.

## Problem Statement

The objective is to build a binary classifier that accepts 15 input features and predicts the `heart_disease` target.

The project addresses several common machine-learning challenges:

- Missing feature values
- Numerical, binary, and categorical input columns
- Class imbalance
- Data-leakage prevention
- Fair comparison of multiple models
- Reusable preprocessing and prediction workflows

## Project Objectives

- Understand the structure and quality of the raw dataset
- Standardise and validate feature values
- Perform exploratory data analysis
- Build reusable preprocessing pipelines
- Train Logistic Regression, Decision Tree, and Random Forest models
- Evaluate performance using metrics suitable for imbalanced classification
- Compare models using the same holdout test set
- Select and save the strongest baseline pipeline
- Reload the saved model and test it on synthetic records
- Document the complete workflow clearly

## Dataset

The cleaned dataset contains:

| Item | Value |
| --- | ---: |
| Total records | 4,238 |
| Total columns | 16 |
| Input features | 15 |
| Target column | `heart_disease` |
| Positive-class proportion | Approximately 15.2% |
| Duplicate rows | 0 |

Dataset files:

- Raw dataset: `data/raw/heart_disease.csv`
- Cleaned dataset: `data/processed/heart_disease_clean.csv`

Missing feature values were deliberately preserved during data cleaning. They are handled later inside the scikit-learn preprocessing pipelines to reduce data leakage and ensure consistent processing.

The dataset source is not explicitly documented in this repository, so no external dataset name is claimed.

## Input Features

### Continuous Numerical Features

| Feature | Description |
| --- | --- |
| `age` | Age of the individual |
| `cigarettes_per_day` | Number of cigarettes smoked per day |
| `total_cholesterol` | Total cholesterol measurement |
| `systolic_bp` | Systolic blood-pressure measurement |
| `diastolic_bp` | Diastolic blood-pressure measurement |
| `bmi` | Body mass index |
| `heart_rate` | Heart-rate measurement |
| `glucose` | Glucose measurement |

### Binary Features

| Feature | Description |
| --- | --- |
| `current_smoker` | Indicates whether the individual currently smokes |
| `bp_meds` | Indicates blood-pressure medication use |
| `prevalent_stroke` | Indicates a previous stroke history |
| `prevalent_hypertension` | Indicates hypertension history |
| `diabetes` | Indicates diabetes status |

### Categorical Features

| Feature | Description |
| --- | --- |
| `gender` | Gender category |
| `education` | Education category |

## Target Variable

The target column is `heart_disease`, where:

- `0` represents no heart disease
- `1` represents heart disease

The positive class represents approximately 15.2% of the dataset. A stratified train-test split was used to preserve approximately the same class distribution in both the training and testing sets.

## Project Workflow

```text
Raw dataset
    ↓
Data understanding
    ↓
Data cleaning and validation
    ↓
Exploratory data analysis
    ↓
Train-test split
    ↓
Feature preprocessing pipelines
    ↓
Logistic Regression baseline
    ↓
Decision Tree and Random Forest
    ↓
Model comparison
    ↓
Final pipeline training and saving
    ↓
Reloaded-model validation
```

## Data Cleaning

The cleaning stage included:

- Standardising column names
- Encoding the target column as `0` and `1`
- Standardising gender values
- Validating binary columns
- Standardising education categories
- Validating numerical values
- Checking for duplicate rows
- Saving the cleaned dataset

No duplicate rows were found.

Missing feature values were not filled during the cleaning phase. They were handled later inside the preprocessing pipelines so that imputation could be learned from training data only.

## Exploratory Data Analysis

The exploratory analysis included:

- Target-class distribution
- Numerical feature distributions
- Boxplots grouped by target class
- Gender-based target rates
- Binary health-indicator comparisons
- Education-based comparisons
- Correlation analysis
- Potential outlier analysis
- Final feature summary

The EDA was used to understand the data structure and support preprocessing and evaluation decisions.

## Preprocessing

The project uses the following scikit-learn components:

- `Pipeline`
- `ColumnTransformer`
- `SimpleImputer`
- `StandardScaler`
- `OneHotEncoder`

### Logistic Regression Preprocessing

- Continuous features: median imputation and standard scaling
- Binary features: most-frequent imputation
- Categorical features: most-frequent imputation and one-hot encoding

### Tree-Model Preprocessing

- Continuous features: median imputation
- Binary features: most-frequent imputation
- Categorical features: most-frequent imputation and one-hot encoding
- No standard scaling

A `ColumnTransformer` applies the appropriate preprocessing steps to each feature group. A `Pipeline` combines preprocessing and the classifier into one reusable object.

During evaluation, preprocessing was fitted using training data only. The test set was transformed using the rules learned from the training set. This reduces data leakage.

## Models

### Logistic Regression

Logistic Regression was used as an interpretable baseline model. It provides probability predictions and a straightforward reference point for model comparison.

### Decision Tree

The Decision Tree model can learn nonlinear decision rules. However, an unrestricted tree may overfit the training data.

### Random Forest

Random Forest combines multiple decision trees. It was included as a more stable ensemble-based tree model.

## Evaluation Metrics

The following metrics were used:

| Metric | Meaning |
| --- | --- |
| Accuracy | Proportion of all predictions that were correct |
| Precision | Proportion of positive predictions that were correct |
| Recall | Proportion of actual positive cases detected |
| F1-score | Balance between precision and recall |
| ROC-AUC | Ability to rank positive cases above negative cases across thresholds |
| PR-AUC | Precision-recall performance for the positive class |

Accuracy alone can be misleading because most records belong to Class 0. PR-AUC was therefore used as the primary model-ranking metric.

## Model Comparison

The following values are holdout testing results:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.8432 | 0.3889 | 0.0543 | 0.0952 | 0.6952 | 0.2937 |
| Random Forest | 0.8396 | 0.2941 | 0.0388 | 0.0685 | 0.6552 | 0.2341 |
| Decision Tree | 0.7311 | 0.1765 | 0.2093 | 0.1915 | 0.5170 | 0.1572 |

The complete comparison report is available at `reports/model_comparison.csv`.

## Selected Model

**Logistic Regression** was selected because it achieved the strongest PR-AUC among the three tested baseline models.

Selected holdout results:

| Metric | Score |
| --- | ---: |
| Accuracy | 0.8432 |
| Precision | 0.3889 |
| Recall | 0.0543 |
| F1-score | 0.0952 |
| ROC-AUC | 0.6952 |
| PR-AUC | 0.2937 |

Although Logistic Regression ranked first according to PR-AUC, its recall at the default `0.50` threshold was only `0.0543`. This means that many positive cases were not detected.

The Decision Tree produced higher recall, but it had substantially weaker ROC-AUC and PR-AUC. Therefore, Logistic Regression should be treated only as the strongest baseline under the defined ranking method.

The selected model is not suitable for medical or clinical use.

## Saved Model

The final saved files are:

- `models/heart_disease_prediction_pipeline.joblib`
- `models/heart_disease_model_metadata.json`

The `.joblib` file contains both:

```text
Preprocessing pipeline
        +
Logistic Regression classifier
```

This allows new records to be provided using the original 15 feature columns. The pipeline automatically applies imputation, scaling, and encoding before generating a prediction.

The `.joblib` file is binary and should not be opened or edited as text.

The model-selection process and final saved artifact are different:

- Holdout testing results were used for honest model comparison
- After selection, the chosen pipeline was retrained using the complete cleaned dataset
- The retrained pipeline was saved for technical demonstration and reproducible prediction testing

## Repository Structure

```text
heart-disease-ml-classification/
├── data/
│   ├── raw/
│   │   └── heart_disease.csv
│   └── processed/
│       └── heart_disease_clean.csv
├── models/
│   ├── heart_disease_model_metadata.json
│   └── heart_disease_prediction_pipeline.joblib
├── notebooks/
│   ├── 01-Data_Understanding.ipynb
│   ├── 02-Data_Cleaning.ipynb
│   ├── 03-EDA.ipynb
│   ├── 04-Preprocessing.ipynb
│   ├── 05-Logistic_Regression.ipynb
│   ├── 06-Tree_Models.ipynb
│   ├── 07-Model_Comparison.ipynb
│   └── 08-Model_Deployment.ipynb
├── reports/
│   ├── final_model_selection.txt
│   └── model_comparison.csv
├── src/
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Notebook Guide

| Notebook | Purpose |
| --- | --- |
| `01-Data_Understanding.ipynb` | Examines dataset shape, columns, data types, missing values, duplicates, class distribution, and feature ranges |
| `02-Data_Cleaning.ipynb` | Standardises and validates features and saves the cleaned dataset |
| `03-EDA.ipynb` | Explores target distribution, feature distributions, correlations, and potential outliers |
| `04-Preprocessing.ipynb` | Defines feature groups, creates a stratified split, and validates preprocessing |
| `05-Logistic_Regression.ipynb` | Builds and evaluates the Logistic Regression baseline |
| `06-Tree_Models.ipynb` | Builds and evaluates Decision Tree and Random Forest models |
| `07-Model_Comparison.ipynb` | Compares holdout performance and selects the strongest baseline |
| `08-Model_Deployment.ipynb` | Retrains, saves, reloads, validates, and tests the selected pipeline |

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- Joblib
- Jupyter Notebook
- Git
- GitHub

## Installation

After cloning or downloading the repository, open Windows PowerShell in the project directory and run:

```powershell
cd heart-disease-ml-classification

python -m venv .venv
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
jupyter notebook
```

When PowerShell is already open inside the repository folder, the `cd` command can be skipped.

## Recommended Notebook Execution Order

Run the notebooks in this order:

```text
01-Data_Understanding.ipynb
02-Data_Cleaning.ipynb
03-EDA.ipynb
04-Preprocessing.ipynb
05-Logistic_Regression.ipynb
06-Tree_Models.ipynb
07-Model_Comparison.ipynb
08-Model_Deployment.ipynb
```

Later notebooks depend on datasets, reports, or model files generated by earlier notebooks.

## Loading the Saved Model and Making a Prediction

The saved Joblib file contains both the preprocessing steps and the trained Logistic Regression classifier.

The following example works when executed from either the project root or the `notebooks` directory:

```python
from pathlib import Path

import joblib
import pandas as pd

current_directory = Path.cwd()

if current_directory.name == "notebooks":
    project_root = current_directory.parent
else:
    project_root = current_directory

model_path = (
    project_root
    / "models"
    / "heart_disease_prediction_pipeline.joblib"
)

if not model_path.exists():
    raise FileNotFoundError(
        f"Model file not found: {model_path}"
    )

loaded_model = joblib.load(model_path)

synthetic_patient = pd.DataFrame([
    {
        "gender": "female",
        "age": 45,
        "education": "graduate",
        "current_smoker": 0,
        "cigarettes_per_day": 0,
        "bp_meds": 0,
        "prevalent_stroke": 0,
        "prevalent_hypertension": 0,
        "diabetes": 0,
        "total_cholesterol": 210,
        "systolic_bp": 125,
        "diastolic_bp": 80,
        "bmi": 24.5,
        "heart_rate": 72,
        "glucose": 85
    }
])

predicted_class = loaded_model.predict(
    synthetic_patient
)[0]

positive_probability = loaded_model.predict_proba(
    synthetic_patient
)[0, 1]

prediction_label = (
    "Heart Disease"
    if predicted_class == 1
    else "No Heart Disease"
)

print("Predicted class:", predicted_class)
print("Prediction label:", prediction_label)
print(
    "Heart-disease probability:",
    round(float(positive_probability), 4)
)
```

The sample record is synthetic and is included only to demonstrate the technical prediction workflow.

Only load `.joblib` files from trusted sources because serialized Python files can execute code when loaded.

## Key Learning Outcomes

This project demonstrates practical experience with:

- Tabular data validation
- Data cleaning
- Exploratory data analysis
- Missing-value handling
- Feature grouping
- One-hot encoding
- Feature scaling
- Train-test splitting
- Stratified sampling
- Scikit-learn pipelines
- `ColumnTransformer`
- Binary-classification metrics
- Class-imbalance evaluation
- Model comparison
- Model persistence using Joblib
- Git-based project development
- Technical documentation

## Limitations

- The dataset contains only 4,238 records
- The positive class is strongly underrepresented
- Dataset quality and sampling bias may affect model performance
- Only three baseline models were compared
- No extensive hyperparameter optimisation was performed
- No class-weight experiments were included in the core project
- The default `0.50` classification threshold was used
- The selected model has very low positive-class recall
- No probability-calibration analysis was performed
- No external validation dataset was used
- No clinical validation was performed
- Coefficients and feature importances represent model associations, not medical causation

## Future Improvements

Possible future extensions include:

- Cross-validated hyperparameter tuning
- Class-weight experiments
- Threshold optimisation
- XGBoost or other gradient-boosting models
- Resampling methods for class imbalance
- Probability calibration
- SHAP-based explainability
- External dataset validation
- Automated tests
- Reusable prediction functions in `src/`
- REST API development
- Web-interface development
- Model monitoring

These are future improvements and are not part of the current completed core project.

## Responsible Use and Medical Disclaimer

> **Important:** This repository is an educational machine-learning portfolio project. It is not a medical device and must not be used for diagnosis, treatment, screening, or clinical decision-making.

Predictions may be incorrect. Anyone with health concerns should seek assessment from a qualified healthcare professional.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.