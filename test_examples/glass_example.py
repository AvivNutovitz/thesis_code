# --- Imports
from test_examples import create_glass_data
from doe_xai import DoeXai
from plotter import Plotter

# thanks to - https://www.kaggle.com/eliekawerk/glass-type-classification-with-machine-learning for the preprocess and model training

# --- Other imports
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import shap
seed = 42


if __name__ == '__main__':
    # --- Data Prepossess
    X_train, y_train, X_test, y_test = create_glass_data()

    print(X_train.columns)
    # --- Model Training
    model = LogisticRegression()
    model.fit(X_train, y_train)
    print("Fitting of Logistic Regression finished")

    xgb_predict = model.predict(X_test)
    xgb_score = accuracy_score(y_test, xgb_predict)
    print(f'test score : {xgb_score}')
    print("=" * 80)
    print(classification_report(y_test, model.predict(X_test)))

    # --- SHAP
    explainer = shap.LinearExplainer(model, X_train)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train, plot_type="bar")

    print(X_train.columns)

    # # --- DOE
    dx = DoeXai(x_data=X_train, y_data=y_train, model=model, feature_names=list(X_train.columns))
    # features: ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'],
    cont = dx.find_feature_contribution(
        user_list=[['RI', 'Na', 'Mg'],
                   ['Ca', 'Ba', 'Fe'],
                   ['Mg', 'Al', 'Si', 'K']])

    # --- Plot
    p = Plotter(X_train)
    p.plot_doe_feature_contribution(cont)