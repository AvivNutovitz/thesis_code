from utils import *
from doe_xai import DoeXai
from plotter import Plotter


# thanks to - https://www.kaggle.com/madz2000/text-classification-using-keras-nb-97-accuracy for the preprocess and model training

# --- imports
import string
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
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


def lemmatize_words(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            pos = pos_tag([i.strip()])
            word = lemmatizer.lemmatize(i.strip(),get_simple_pos(pos[0][1]))
            final_text.append(word.lower())
    return " ".join(final_text)


# --- start here
X, y = load_data('fake_job_posting', size=1000)

# --- Data Prepossess
stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)
lemmatizer = WordNetLemmatizer()
X = X.apply(lemmatize_words)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

tv = TfidfVectorizer(min_df=0, max_df=1, use_idf=True, ngram_range=(1, 3), max_features=20)
tv_train_reviews = tv.fit_transform(X_train)
tv_test_reviews = tv.transform(X_test)

# --- Model Training
mnb = MultinomialNB()
mnb_tfidf = mnb.fit(tv_train_reviews, y_train)

mnb_tfidf_predict = mnb.predict(tv_test_reviews)
mnb_tfidf_score = accuracy_score(y_test, mnb_tfidf_predict)
print(f'mnb tfidf test score : {mnb_tfidf_score}')


# --- DOE
dx = DoeXai(x_data=tv_train_reviews, y_data=y_train, model=mnb, feature_names=tv.get_feature_names())

cont = dx.find_feature_contribution()
print(cont)

# --- Plot
# p = Plotter(tv_train_reviews.toarray(), tv.get_feature_names())
# p.plot_doe_feature_contribution(cont)


