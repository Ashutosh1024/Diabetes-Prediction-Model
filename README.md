# Diabetes Prediction Using Logistic Regression

## Project Overview

This project builds a machine learning model to predict diabetes presence based on medical diagnostic data. Using Logistic Regression, the model classifies individuals as diabetic or non-diabetic by analyzing key health indicators. The workflow includes data preprocessing, exploratory data analysis, visualization, model training, and evaluation.


## Dataset

The dataset used is the [Pima Indians Diabetes Dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database), containing medical measurements from female Pima Indian patients.

**Features include:**

* Pregnancies
* Glucose
* BloodPressure
* SkinThickness
* Insulin
* BMI (Body Mass Index)
* DiabetesPedigreeFunction
* Age
* Outcome (0 = Non-Diabetic, 1 = Diabetic)


## Exploratory Data Analysis & Visualization

* **Class Distribution:**
  Visualized using a count plot to understand the balance between diabetic and non-diabetic cases.

* **Correlation Heatmap:**
  Displays correlations between features to identify important relationships and multicollinearity.


## Data Preprocessing

* Invalid zero values in critical columns (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) were replaced with NaN and imputed using the median to handle missing data appropriately.

* Data was split into training and testing subsets, then features were standardized using `StandardScaler` for optimal model performance.


## Model Training & Evaluation

* Logistic Regression was employed as the classification algorithm.

* The model was trained on scaled training data and evaluated on test data.

* Performance metrics include accuracy, classification report (precision, recall, F1-score), and confusion matrix.


## Evaluation Metrics

* **Accuracy:** Measures overall correctness of the model predictions.

* **Classification Report:** Provides detailed precision, recall, and F1-score per class.

* **Confusion Matrix:** Visualizes true vs predicted classifications, highlighting false positives and false negatives.


## Results

* The model achieves strong predictive performance, with accuracy and detailed metrics displayed in the output.

* Confusion matrix visualization helps identify where the model succeeds or misclassifies.


## Future Directions

* Explore alternative classification algorithms such as Random Forest, Support Vector Machines, or Gradient Boosting.

* Implement hyperparameter tuning to optimize model parameters.

* Investigate feature engineering and selection to improve predictive power.

* Address any class imbalance through resampling or synthetic data generation methods if necessary.


## License

This project is open-source and free for educational and research use.
