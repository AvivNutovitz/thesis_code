# --- Imports
from doe_utils import load_data
from doe_xai import DoeXai
from plotter import Plotter

# thanks to - https://www.kaggle.com/madz2000/text-classification-using-keras-nb-97-accuracy for the preprocess and model training

# --- Other imports
import string
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pandas as pd
import shap
seed = 42


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


def create_fake_job_posting_data_and_tv(size=2000):

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


if __name__ == '__main__':
    # --- Data Prepossess
    train_reviews, y_train, test_reviews, y_test, tv = create_fake_job_posting_data_and_tv()

    # --- Model Training
    mnb = MultinomialNB()
    mnb_tfidf = mnb.fit(train_reviews, y_train)

    mnb_tfidf_predict = mnb.predict(test_reviews)
    mnb_tfidf_score = accuracy_score(y_test, mnb_tfidf_predict)
    print(f'mnb tfidf test score : {mnb_tfidf_score}')


    # --- SHAP
    train_clean_data = pd.DataFrame(train_reviews.toarray(), columns=tv.get_feature_names())
    explainer = shap.LinearExplainer(mnb, train_clean_data)
    shap_values = explainer.shap_values(train_clean_data)
    shap.summary_plot(shap_values, train_clean_data, plot_type="bar")


    # --- DOE
    dx = DoeXai(x_data=train_clean_data, y_data=y_train, model=mnb, feature_names=tv.get_feature_names())
    # features to test: ['echoing', 'echoing green', 'epsilon', 'iota', 'lambda', 'omicron', 'pi', 'rho', 'sigmaf', 'tau']
    cont = dx.find_feature_contribution(only_orig_features=True)
    print(cont)


    # --- Plot
    p = Plotter(train_clean_data)
    p.plot_doe_feature_contribution(cont)


