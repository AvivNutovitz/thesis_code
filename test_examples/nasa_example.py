# --- Imports
from test_examples import create_nasa_data
from doe_xai import DoeXai
from plotter import Plotter

# thanks to - https://www.kaggle.com/winvoker/asteroid-classification-99-36-acc for the preprocess and model training

# --- Other imports
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import shap
seed = 42


if __name__ == '__main__':
    # --- Data Prepossess
    X_train, y_train, X_test, y_test = create_nasa_data()

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
    # features: ['Absolute_Magnitude', 'Est_Dia_in_KM(min)', 'Est_Dia_in_KM(max)',
    #        'Est_Dia_in_M(min)', 'Est_Dia_in_M(max)', 'Est_Dia_in_Miles(min)',
    #        'Est_Dia_in_Miles(max)', 'Est_Dia_in_Feet(min)', 'Est_Dia_in_Feet(max)',
    #        'Relative_Velocity_km_per_sec', 'Relative_Velocity_km_per_hr',
    #        'Miles_per_hour', 'Miss_Dist.(Astronomical)', 'Miss_Dist.(lunar)',
    #        'Miss_Dist.(kilometers)', 'Miss_Dist.(miles)', 'Orbit_ID',
    #        'Orbit_Uncertainity', 'Minimum_Orbit_Intersection',
    #        'Jupiter_Tisserand_Invariant', 'Epoch_Osculation', 'Eccentricity',
    #        'Semi_Major_Axis', 'Inclination', 'Asc_Node_Longitude',
    #        'Orbital_Period', 'Perihelion_Distance', 'Perihelion_Arg',
    #        'Aphelion_Dist', 'Perihelion_Time', 'Mean_Anomaly', 'Mean_Motion'],
    cont = dx.find_feature_contribution(user_list=[['Absolute_Magnitude',
                                                   'Est_Dia_in_KM(min)', 'Est_Dia_in_KM(max)']])

    # --- Plot
    p = Plotter(X_train)
    p.plot_doe_feature_contribution(cont)