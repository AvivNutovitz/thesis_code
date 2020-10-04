from utils import *
from doe_xai import DoeXai
from plotter import Plotter

# thanks to - https://www.kaggle.com/marcuswingen/eda-of-bookings-and-ml-to-predict-cancelations for the preprocess and model training

# --- imports
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
seed = 42


# --- helper function
def get_clean_enc_cat_features(enc_cat_f, cat_f):
    rv = []
    for enc_cat in enc_cat_f:
        idx = enc_cat.split('_')[0].split('x')[1]
        feature_value = enc_cat.split('_')[1]
        rv.append((cat_f[int(idx)]+'_'+feature_value))
    return rv


def create_hotel_booking_data(size=2000):
    X, y = load_data('hotel_bookings', size=size)

    num_features = ["lead_time", "arrival_date_week_number", "arrival_date_day_of_month",
                    "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children",
                    "babies", "is_repeated_guest", "previous_cancellations",
                    "previous_bookings_not_canceled", "agent", "company",
                    "required_car_parking_spaces", "total_of_special_requests", "adr"]

    cat_features = ["hotel", "arrival_date_month", "meal", "market_segment",
                    "distribution_channel", "reserved_room_type", "deposit_type", "customer_type"]
    features = num_features + cat_features
    X = X[features]
    # Separate features and predicted value
    features = num_features + cat_features
    num_transformer = SimpleImputer(strategy="constant")
    # Preprocessing for categorical features:
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown='ignore'))])
    # Bundle preprocessing for numerical and categorical features:
    preprocessor = ColumnTransformer(transformers=[("num", num_transformer, num_features),
                                                   ("cat", cat_transformer, cat_features)], remainder='drop')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    X_train = preprocessor.fit_transform(X_train)
    enc_cat_features = preprocessor.named_transformers_['cat']['onehot'].get_feature_names()
    clean_enc_cat_features = get_clean_enc_cat_features(enc_cat_features, cat_features)
    labels = np.concatenate([num_features, clean_enc_cat_features])
    X_train = pd.DataFrame(X_train, columns=labels)
    X_test = pd.DataFrame(preprocessor.transform(X_test), columns=labels)

    return X_train, y_train, X_test, y_test, labels


# --- Data Prepossess
X_train, y_train, X_test, y_test, labels = create_hotel_booking_data()

# --- Model Training
model = LogisticRegression(random_state=seed, n_jobs=-1)
model.fit(X_train, y_train)
preds = model.predict_proba(X_test)
score = roc_auc_score(y_test, [p[1] for p in preds])
print(f"auc_score: {round(score, 4)}")


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