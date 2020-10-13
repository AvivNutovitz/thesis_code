# --- Imports
from test_examples import create_placement_full_class_data
from doe_xai import DoeXai
from plotter import Plotter

# thanks to - https://www.kaggle.com/vinayshaw/will-you-get-a-job-or-not-eda-prediction for the prepossess and model training

# --- Other imports
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import shap
seed = 42


if __name__ == '__main__':
    # --- Data Prepossess
    X_train, y_train, X_test, y_test = create_placement_full_class_data()

    # --- Model Training
    model = LogisticRegression(random_state=seed)
    model.fit(X_train, y_train)

    test_score = model.score(X_test, y_test)
    print(f"model score: {test_score}")
    print("=" * 80)
    print(classification_report(y_test, model.predict(X_test)))

    # --- SHAP
    explainer = shap.LinearExplainer(model, X_train)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train, plot_type="bar")

    # --- DOE
    dx = DoeXai(x_data=X_train, y_data=y_train, model=model, feature_names=list(X_train.columns))
    # features: ['ssc_p', 'mba_p', 'hsc_p', 'degree_p', 'degree_t', 'workex', 'gender', 'etest_p', 'hsc_p', 'hsc_b',
    # 'specialisation']
    cont = dx.find_feature_contribution(user_list=[['ssc_p', 'mba_p'], ['hsc_p', 'degree_p', 'degree_t']])
    print(cont)

    # --- Plot
    p = Plotter(X_train)
    p.plot_doe_feature_contribution(cont)