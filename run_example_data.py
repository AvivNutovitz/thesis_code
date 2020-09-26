import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine

from doe_xai import DoeXai


seed = 42

# read in the iris data
wine = load_wine()

# create X (features) and y (response)
X = wine.data
y = wine.target

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)

# build model
model = LogisticRegression(random_state=seed)

# fit model
model.fit(X_train, y_train)

# get test score
test_score = model.score(X_test, y_test)

dx = DoeXai(x_data=X_train, y_data=y_train, model=model, feature_names=wine.feature_names)

cont = dx.find_feature_contribution(user_list=[[1, 2, 3, 4], [1, 2, 3], [5, 6]])
# print(dx.all_predictions_df.mean(axis=1))
#
# print(dx.all_predictions_df)
#
# print(dx.zeds_df)

print(cont)
