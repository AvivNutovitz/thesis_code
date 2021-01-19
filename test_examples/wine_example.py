# --- Imports
from test_examples import create_wine_data
from doe_xai import DoeXai
from plotter import Plotter
from doe_utils import *

# --- Other imports
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import shap
from numpy import random
seed = 42


if __name__ == '__main__':
    # --- Data Prepossess
    X_train, y_train, X_test, y_test = create_wine_data()

    # --- Model Training
    model = LogisticRegression(random_state=random.seed(seed))
    model.fit(X_train, y_train)
    print("Fitting of Logistic Regression Classifier finished")

    rf_predict = model.predict(X_test)
    rf_score = accuracy_score(y_test, rf_predict)
    print(f'test score : {rf_score}')
    print("=" * 80)
    print(classification_report(y_test, model.predict(X_test)))

    # --- SHAP
    explainer = shap.LinearExplainer(model, X_train)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train, plot_type="bar")

    # --- DOE
    dx = DoeXai(x_data=X_train, y_data=y_train, model=model, feature_names=list(X_train.columns))
    # features:
    """    
    ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids',
    'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']
    """
    cont = dx.find_feature_contribution(user_list=[['nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue'],
                                                   ['alcohol', 'malic_acid', 'ash'], ['hue', 'proline']])

    # --- Plot
    p = Plotter(X_train)
    p.plot_doe_feature_contribution(cont)

    # --- Add feature interaction and re train model
    X_train['alcohol_malic_acid_ash'] = X_train['alcohol'] * X_train['malic_acid'] * X_train['ash']
    X_test['alcohol_malic_acid_ash'] = X_test['alcohol'] * X_test['malic_acid'] * X_test['ash']

    model = LogisticRegression(random_state=random.seed(seed))
    model.fit(X_train, y_train)
    print("Fitting of Logistic Regression Classifier finished  with new feature base on interaction")

    rf_predict = model.predict(X_test)
    rf_score = accuracy_score(y_test, rf_predict)
    print(f'test score : {rf_score}')
    print("=" * 80)
