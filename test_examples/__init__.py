# --- Imports
from doe_utils import load_data

# --- Other imports
import string
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
seed = 42


def create_fake_job_posting_data_and_tv(size=2000):
    # thanks to - https://www.kaggle.com/madz2000/text-classification-using-keras-nb-97-accuracy
    # for the prepossess and model training

    # --- helper functions
    def get_simple_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def lemmatize_words(text):
        final_text = []
        for i in text.split():
            if i.strip().lower() not in stop:
                pos = pos_tag([i.strip()])
                word = lemmatizer.lemmatize(i.strip(), get_simple_pos(pos[0][1]))
                final_text.append(word.lower())
        return " ".join(final_text)

    X, y = load_data('fake_job_posting', size=size)

    stop = set(stopwords.words('english'))
    punctuation = list(string.punctuation)
    stop.update(punctuation)
    lemmatizer = WordNetLemmatizer()
    X = X.apply(lemmatize_words)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

    tv = TfidfVectorizer(min_df=0, max_df=1, use_idf=True, ngram_range=(1, 3), max_features=10)
    tv_train_reviews = tv.fit_transform(X_train)
    tv_test_reviews = tv.transform(X_test)
    return tv_train_reviews, y_train, tv_test_reviews, y_test, tv


def create_hotel_booking_data(size=2000):
    # thanks to - https://www.kaggle.com/marcuswingen/eda-of-bookings-and-ml-to-predict-cancelations
    # for the prepossess and model training

    # --- helper functions
    def get_clean_enc_cat_features(enc_cat_f, cat_f):
        rv = []
        for enc_cat in enc_cat_f:
            idx = enc_cat.split('_')[0].split('x')[1]
            feature_value = enc_cat.split('_')[1]
            rv.append((cat_f[int(idx)] + '_' + feature_value))
        return rv

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


def create_wine_data(size=-1):
    X, y = load_data('wine', size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
    return X_train, y_train, X_test, y_test


def create_hr_employee_attrition_data(size=-1):
    # thanks to -  https://www.kaggle.com/arthurtok/employee-attrition-via-ensemble-tree-based-methods
    # for the prepossess and model training

    X, y = load_data('hr_employee_attrition', size)

    categorical = []
    for col, value in X.iteritems():
        if value.dtype == 'object':
            categorical.append(col)

    # Store the numerical columns in a list numerical
    numerical = X.columns.difference(categorical)

    x_cat = X[categorical]
    x_cat = pd.get_dummies(x_cat)
    x_num = X[numerical]
    x_final = pd.concat([x_num, x_cat], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(x_final, y, random_state=seed)
    return X_train, y_train, X_test, y_test


def create_nomao_data(size=10000):
    X, y = load_data('nomao', size)
    clean_y = np.array([int(yy) - 1 for yy in y])

    X_train, X_test, y_train, y_test = train_test_split(X, clean_y, random_state=seed)
    return X_train, y_train, X_test, y_test


def create_placement_full_class_data(size=-1):
    # thanks to - https://www.kaggle.com/vinayshaw/will-you-get-a-job-or-not-eda-prediction
    # for the prepossess and model training

    X, y = load_data('placement_full_class', size)

    X["gender"] = X.gender.map({"M": 0, "F": 1})
    X["ssc_b"] = X.ssc_b.map({"Others": 0, "Central": 1})
    X["hsc_b"] = X.hsc_b.map({"Others": 0, "Central": 1})
    X["hsc_s"] = X.hsc_s.map({"Commerce": 0, "Science": 1, "Arts": 2})
    X["degree_t"] = X.degree_t.map({"Comm&Mgmt": 0, "Sci&Tech": 1, "Others": 2})
    X["workex"] = X.workex.map({"No": 0, "Yes": 1})
    X["specialisation"] = X.specialisation.map({"Mkt&HR": 0, "Mkt&Fin": 1})

    y = y.map({"Not Placed": 0, "Placed": 1})
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
    return X_train, y_train, X_test, y_test

