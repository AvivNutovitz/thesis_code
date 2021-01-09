# --- Imports
from test_examples import create_nomao_data
from doe_xai import DoeXai
from plotter import Plotter

# --- Other imports
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import shap
from numpy import random
seed = 42


if __name__ == '__main__':
    # --- Data Prepossess
    X_train, y_train, X_test, y_test = create_nomao_data()

    # --- Model Training
    model = LogisticRegression(random_state=random.seed(seed))
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

    # features: V1 -> V118
    cont = dx.find_feature_contribution(only_orig_features=True)
    print(cont)

    # --- Plot
    p = Plotter(X_train)
    p.plot_doe_feature_contribution(cont)