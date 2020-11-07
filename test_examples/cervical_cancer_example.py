# --- Imports
from test_examples import create_cervical_cancer_data
from doe_xai import DoeXai
from plotter import Plotter

# thanks to - https://www.kaggle.com/niyamatalmass/ml-explainability-deep-dive-into-the-ml-model for the preprocess and model training

# --- Other imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import shap
seed = 42


if __name__ == '__main__':
    # --- Data Prepossess
    X_train, y_train, X_test, y_test = create_cervical_cancer_data()

    # --- Model Training
    model = RandomForestClassifier(n_estimators=500, random_state=seed)
    model.fit(X_train, y_train)
    print("Fitting of Random Forest Classifier finished")

    xgb_predict = model.predict(X_test)
    xgb_score = accuracy_score(y_test, xgb_predict)
    print(f'test score : {xgb_score}')
    print("=" * 80)
    print(classification_report(y_test, model.predict(X_test)))

    # --- SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train, plot_type="bar")

    print(X_train.columns)

    # # --- DOE
    dx = DoeXai(x_data=X_train, y_data=y_train, model=model, feature_names=list(X_train.columns))
    # features: ['Age', 'Number of sexual partners', 'First sexual intercourse',
    #        'Num of pregnancies', 'Smokes', 'Smokes (years)', 'Smokes (packs/year)',
    #        'Hormonal Contraceptives', 'Hormonal Contraceptives (years)', 'IUD',
    #        'IUD (years)', 'STDs', 'STDs (number)', 'STDs:condylomatosis',
    #        'STDs:cervical condylomatosis', 'STDs:vaginal condylomatosis',
    #        'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis',
    #        'STDs:pelvic inflammatory disease', 'STDs:genital herpes',
    #        'STDs:molluscum contagiosum', 'STDs:AIDS', 'STDs:HIV',
    #        'STDs:Hepatitis B', 'STDs:HPV', 'STDs: Number of diagnosis',
    #        'STDs: Time since first diagnosis', 'STDs: Time since last diagnosis',
    #        'Dx:Cancer', 'Dx:CIN', 'Dx:HPV', 'Dx', 'Hinselmann', 'Schiller',
    #        'Citology'],
    cont = dx.find_feature_contribution(
        user_list=[['STDs: Time since first diagnosis', 'Age', 'Smokes', 'Smokes (years)'],
                   ['STDs: Number of diagnosis', 'STDs: Time since first diagnosis',
                    'STDs: Time since last diagnosis']])

    # --- Plot
    p = Plotter(X_train)
    p.plot_doe_feature_contribution(cont)