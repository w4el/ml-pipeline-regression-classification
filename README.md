# End-to-End Machine Learning Pipelines

> A project demonstrating the implementation of complete, robust, and reproducible machine learning workflows for both regression and classification tasks.
> * **Regression Task:** Predicting California housing prices.
> * **Classification Task:** Predicting passenger survival on the Titanic.

The core focus of this project is not just model training, but building a systematic and reusable end-to-end pipeline, from raw data ingestion and cleaning to final model evaluation and hyperparameter tuning.

## 1. Key Features & Technical Implementation

This project is built almost entirely using Scikit-learn, Pandas, and Matplotlib, demonstrating a mastery of the core data science stack.

### Advanced Pre-processing Pipeline
A single, robust pre-processing pipeline was constructed using Scikit-learn's `Pipeline` and `ColumnTransformer`. This pipeline automatically applies the correct transformations to the correct data types:
* **Missing Values:** `SimpleImputer` is used to fill in missing data (e.g., `total_bedrooms`).
* **Numerical Features:** `StandardScaler` is applied to all numerical data to normalise its range.
* **Categorical Features:** `OneHotEncoder` is used to convert categorical data into a machine-readable format.

### Regression Task (California Housing)
* **Models Implemented:** Linear Regression, Decision Tree Regression, and **Random Forest Regression**.
* **Evaluation Metrics:** Models were rigorously evaluated using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² score.
* **Tuning:** `GridSearchCV` was employed to systematically find the optimal hyperparameters for the Random Forest model.

### Classification Task (Titanic Survival)
* **Models Implemented:** Logistic Regression, **Random Forest Classifier**, and **Gradient Boosting Classifier**.
* **Evaluation Metrics:** Due to the nature of the survival dataset, models were evaluated using a full suite of metrics: Accuracy, Precision, Recall, and **F1-Score**.
* **Tuning:** `GridSearchCV` was again used to tune the ensemble models for maximum predictive accuracy.

## 2. Technologies & Libraries Used

* **Core:** Python
* **Data Manipulation:** Pandas, NumPy
* **ML Pipeline & Modelling:** Scikit-learn (extensively used for `Pipeline`, `ColumnTransformer`, `GridSearchCV`, and all models)
* **Data Visualisation:** Matplotlib, Seaborn
* **Environment:** Jupyter Notebook

## 3. Installation & Usage

1.  Clone the repository:
    `git clone https://github.com/w4el/ml-pipeline-regression-classification.git`
2.  Navigate to the project directory:
    `cd ml-pipeline-regression-classification`
3.  Install the required dependencies from the `requirements.txt` file:
    `pip install -r requirements.txt`

The entire analysis—from data loading, pipeline construction, model training, and evaluation—is contained within the Jupyter Notebook.
1.  Launch Jupyter:
    `jupyter notebook`
2.  Open the `ml-pipeline-regression-classification.ipynb` file and run all cells to replicate the results.
