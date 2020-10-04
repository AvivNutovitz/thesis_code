# --- imports
from test_examples import create_hotel_booking_data
from doe_xai import DoeXai
from plotter import Plotter

# thanks to - https://www.kaggle.com/marcuswingen/eda-of-bookings-and-ml-to-predict-cancelations for the preprocess and model training

# --- Other imports
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

import shap
seed = 42

if __name__ == '__main__':
    # --- Data Prepossess
    X_train, y_train, X_test, y_test, labels = create_hotel_booking_data()

    # --- Model Training
    model = LogisticRegression(random_state=seed, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)
    score = roc_auc_score(y_test, [p[1] for p in preds])
    print(f"auc_score: {round(score, 4)}")
    print("=" * 80)
    print(classification_report(y_test, model.predict(X_test)))

    # --- SHAP
    explainer = shap.LinearExplainer(model, X_train)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train, plot_type="bar")

    # --- DOE
    dx = DoeXai(x_data=X_train, y_data=y_train, model=model, feature_names=labels)
    features_to_test = [['company', 'agent'], ['company', 'agent', 'children']]
    cont = dx.find_feature_contribution(user_list=features_to_test)
    print(cont)

    # --- Plot
    p = Plotter(X_train)
    p.plot_doe_feature_contribution(cont)