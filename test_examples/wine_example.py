# --- Imports
from utils import *
from doe_xai import DoeXai
from plotter import Plotter

# --- Other imports
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
seed = 42

# --- Start Here
X, y = load_data('wine')

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)

# --- Model Training
model = LogisticRegression(random_state=seed)
model.fit(X_train, y_train)

test_score = model.score(X_test, y_test)
print(test_score)

# --- SHAP
explainer = shap.LinearExplainer(model, X_train)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train, plot_type="bar")

# --- DOE
dx = DoeXai(x_data=X_train, y_data=y_train, model=model, feature_names=list(X.columns))
# features: ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids',
# 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']
cont = dx.find_feature_contribution(user_list=[['nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue'],
                                               ['alcohol', 'malic_acid', 'ash'], ['hue', 'proline']])
print(cont)

# --- Plot
p = Plotter(X_train)
p.plot_doe_feature_contribution(cont)