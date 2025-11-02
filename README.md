# End-to-End Machine Learning Pipelines

> This project demonstrates the implementation of complete machine learning workflows for both regression and classification tasks.
> * **Regression Task:** Predicting California housing prices.
> * **Classification Task:** Predicting passenger survival on the Titanic.

The core of this project is building robust, reproducible pre-processing and modelling pipelines using Scikit-learn, from raw data ingestion to final model evaluation and tuning.

## 1. Key Features

* **Full Pre-processing Pipeline:** Uses Scikit-learn's `Pipeline` and `ColumnTransformer` to systematically handle missing values (`SimpleImputer`), scale numerical data (`StandardScaler`), and encode categorical data (`OneHotEncoder`).
* **Model Implementation & Comparison:**
    * **Regression:** Implements and compares Linear Regression, Decision Tree Regression, and Random Forest Regression.
    * **Classification:** Implements and compares Logistic Regression, Random Forest Classifier, and Gradient Boosting Classifier.
* **Systematic Hyperparameter Tuning:** Employs `GridSearchCV` to exhaustively search parameter grids and find the optimal hyperparameters for the best-performing models.
* **Robust Evaluation:**
    * Regression models are evaluated using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² score.
    * Classification models are evaluated using Accuracy, Precision, Recall, and F1-Score.

## 2. Technologies & Libraries Used

* **Core:** Python, Jupyter Notebook
* **Data Science:** Scikit-learn (for all pipelines, models, and metrics)
* **Data Manipulation:** Pandas, NumPy
* **Data Visualisation:** Matplotlib, Seaborn

## 3. Installation & Usage

1.  Clone the repository:
    `git clone https://github.com/w4el/ml-pipeline-regression-classification.git`
2.  Navigate to the project directory:
    `cd ml-pipeline-regression-classification`
3.  Install the required dependencies (you must create this `requirements.txt` file):
    `pip install -r requirements.txt`

The entire analysis, from pre-processing to model tuning and evaluation, is contained in the Jupyter Notebook.
1.  Launch Jupyter:
    `jupyter notebook`
2.  Open the `COMP1816_CW_Code_Notebook.ipynb` file and run all cells.
