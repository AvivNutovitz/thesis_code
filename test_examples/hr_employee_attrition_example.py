# --- Imports
from test_examples import create_hr_employee_attrition_data
from doe_xai import DoeXai
from plotter import Plotter

# thanks to - https://www.kaggle.com/madz2000/text-classification-using-keras-nb-97-accuracy for the preprocess and model training

# --- Other imports
import shap
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report
from numpy import random
seed = 42


if __name__ == '__main__':

    # --- Data Prepossess
    X_train, y_train, X_test, y_test = create_hr_employee_attrition_data()

    # --- Model Training
    model = GradientBoostingClassifier(n_estimators=500, random_state=random.seed(seed))
    model.fit(X_train, y_train)
    print("Fitting of Gradient Boosting Classifier finished")
    rf_probs = model.predict_proba(X_test)
    rf_predictions = model.predict(X_test)

    score = roc_auc_score(y_test, [p[1] for p in rf_probs])
    print(f"roc_auc score: {score}")
    print("=" * 80)
    print(classification_report(y_test, rf_predictions))

    # --- SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train, plot_type="bar")

    # --- DOE
    dx = DoeXai(x_data=X_train, y_data=y_train, model=model, feature_names=list(X_train.columns))
    # features:
    """
    ['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EmployeeCount',
       'EmployeeNumber', 'EnvironmentSatisfaction', 'HourlyRate',
       'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome',
       'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike',
       'PerformanceRating', 'RelationshipSatisfaction', 'StandardHours',
       'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
       'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
       'YearsSinceLastPromotion', 'YearsWithCurrManager',
       'BusinessTravel_Non-Travel', 'BusinessTravel_Travel_Frequently',
       'BusinessTravel_Travel_Rarely', 'Department_Human Resources',
       'Department_Research & Development', 'Department_Sales',
       'EducationField_Human Resources', 'EducationField_Life Sciences',
       'EducationField_Marketing', 'EducationField_Medical',
       'EducationField_Other', 'EducationField_Technical Degree',
       'Gender_Female', 'Gender_Male', 'JobRole_Healthcare Representative',
       'JobRole_Human Resources', 'JobRole_Laboratory Technician',
       'JobRole_Manager', 'JobRole_Manufacturing Director',
       'JobRole_Research Director', 'JobRole_Research Scientist',
       'JobRole_Sales Executive', 'JobRole_Sales Representative',
       'MaritalStatus_Divorced', 'MaritalStatus_Married',
       'MaritalStatus_Single', 'Over18_Y', 'OverTime_No', 'OverTime_Yes']"""
    cont = dx.find_feature_contribution(only_orig_features=True)
    print(cont)
    print(X_train.columns)

    # --- Plot
    p = Plotter(X_train)
    p.plot_doe_feature_contribution(cont)


