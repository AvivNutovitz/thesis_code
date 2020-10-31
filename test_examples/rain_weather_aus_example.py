# --- Imports
from test_examples import create_rain_weather_aus
from doe_xai import DoeXai
from doe_utils import t_test_over_doe_shap_differences

from plotter import Plotter

# thanks to - https://www.kaggle.com/aninditapani/will-it-rain-tomorrow for the prepossess and model training

# --- Other imports
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
import shap
seed = 42


if __name__ == '__main__':
    # --- Data Prepossess
    X_train, y_train, X_test, y_test = create_rain_weather_aus()

    # --- Model Training
    model = GradientBoostingClassifier(n_estimators=500, random_state=seed)
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
    # shap.summary_plot(shap_values, X_train, plot_type="bar")

    # --- DOE
    dx = DoeXai(x_data=X_train, y_data=y_train, model=model, feature_names=list(X_train.columns))

    # features: ['MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
    # 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm', 'RainToday', 'WindGustDir_E',
    # 'WindGustDir_ENE', 'WindGustDir_ESE', 'WindGustDir_N', 'WindGustDir_NE', 'WindGustDir_NNE', 'WindGustDir_NNW',
    # 'WindGustDir_NW', 'WindGustDir_S', 'WindGustDir_SE', 'WindGustDir_SSE', 'WindGustDir_SSW', 'WindGustDir_SW',
    # 'WindGustDir_W', 'WindGustDir_WNW', 'WindGustDir_WSW', 'WindDir3pm_E', 'WindDir3pm_ENE', 'WindDir3pm_ESE',
    # 'WindDir3pm_N', 'WindDir3pm_NE', 'WindDir3pm_NNE', 'WindDir3pm_NNW', 'WindDir3pm_NW', 'WindDir3pm_S',
    # 'WindDir3pm_SE', 'WindDir3pm_SSE', 'WindDir3pm_SSW', 'WindDir3pm_SW', 'WindDir3pm_W', 'WindDir3pm_WNW',
    # 'WindDir3pm_WSW', 'WindDir9am_E', 'WindDir9am_ENE', 'WindDir9am_ESE', 'WindDir9am_N', 'WindDir9am_NE',
    # 'WindDir9am_NNE', 'WindDir9am_NNW', 'WindDir9am_NW', 'WindDir9am_S', 'WindDir9am_SE', 'WindDir9am_SSE',
    # 'WindDir9am_SSW', 'WindDir9am_SW', 'WindDir9am_W', 'WindDir9am_WNW', 'WindDir9am_WSW']
    # cont = dx.find_feature_contribution(user_list=[['MinTemp', 'MaxTemp', 'Rainfall'], ['Rainfall', 'WindGustSpeed'],
    #                                                ['Temp9am', 'Temp3pm', 'RainToday', 'WindGustDir_E']])
    # print(cont)

    cont = dx.find_feature_contribution(only_orig_features=True)

    # t_stat, pvalue = t_test_over_doe_shap_differences(shap_values, cont, X_train.columns, do_random=True)
    # print(pvalue)
    #
    # t_stat, pvalue = t_test_over_doe_shap_differences(shap_values, cont, X_train.columns, do_random=False)
    # print(pvalue)

    # --- Plot
    # p = Plotter(X_train)
    # p.plot_doe_feature_contribution(cont)