# --- Imports
from test_examples import create_fake_job_posting_data_and_tv
from doe_xai import DoeXai
from plotter import Plotter

# thanks to - https://www.kaggle.com/madz2000/text-classification-using-keras-nb-97-accuracy for the preprocess and model training

# --- Other imports
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import shap
seed = 42


if __name__ == '__main__':
    # --- Data Prepossess
    train_clean_reviews, y_train, test_clean_reviews, y_test = create_fake_job_posting_data_and_tv()

    # --- Model Training
    mnb = MultinomialNB()
    mnb_tfidf = mnb.fit(train_clean_reviews, y_train)

    mnb_tfidf_predict = mnb.predict(test_clean_reviews)
    mnb_tfidf_score = accuracy_score(y_test, mnb_tfidf_predict)
    print(f'mnb tfidf test score : {mnb_tfidf_score}')
    print("=" * 80)
    print(classification_report(y_test, mnb_tfidf_predict))

    # --- SHAP
    explainer = shap.LinearExplainer(mnb, train_clean_reviews)
    shap_values = explainer.shap_values(train_clean_reviews)
    shap.summary_plot(shap_values, train_clean_reviews, plot_type="bar")

    # --- DOE
    dx = DoeXai(x_data=train_clean_reviews, y_data=y_train, model=mnb)
    # features to test: ['echoing', 'echoing green', 'epsilon', 'iota', 'lambda', 'omicron', 'pi', 'rho', 'sigmaf', 'tau']
    cont = dx.find_feature_contribution(only_orig_features=True)
    print(cont)

    # --- Plot
    p = Plotter(train_clean_reviews)
    p.plot_doe_feature_contribution(cont)


