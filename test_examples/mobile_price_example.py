# --- Imports
from test_examples import create_mobile_price_data
from doe_xai import DoeXai
from plotter import Plotter

# --- Other imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import shap
from numpy import random
seed = 42


if __name__ == '__main__':
    # --- Data Prepossess
    X_train, y_train, X_test, y_test = create_mobile_price_data()

    # --- Model Training
    model = RandomForestClassifier(n_estimators=500, random_state=random.seed(seed))
    model.fit(X_train, y_train)
    print("Fitting of Random Forest Classifier finished")

    rf_predict = model.predict(X_test)
    rf_score = accuracy_score(y_test, rf_predict)
    print(f'test score : {rf_score}')
    print("=" * 80)
    print(classification_report(y_test, model.predict(X_test)))

    # --- SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train, plot_type="bar")

    # # --- DOE
    dx = DoeXai(x_data=X_train, y_data=y_train, model=model, feature_names=list(X_train.columns))
    # features:
    """['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
    'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w',
    'talk_time', 'three_g', 'touch_screen', 'wifi']"""
    cont = dx.find_feature_contribution(user_list=[['blue', 'clock_speed', 'dual_sim'],
                                                   ['px_height', 'px_width', 'ram', 'sc_h', 'sc_w'],
                                                   ['int_memory', 'm_dep', 'mobile_wt']])

    # --- Plot
    p = Plotter(X_train)
    p.plot_doe_feature_contribution(cont)