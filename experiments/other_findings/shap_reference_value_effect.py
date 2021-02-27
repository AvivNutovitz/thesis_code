from experiments.get_datasets import *
from doe_xai_utils import *
import shap


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = create_wine_data()
    print("finish load data hotel_booking")

    model = tabular_cnn(len(list(X_train.columns)), len(set(y_train)))
    model_name = 'DNN'

    X_train_, y_train_ = set_deep_model_data(X_train, y_train)

    size = 50
    print(f"train DeepExplainer on {model_name}")
    explainer = shap.DeepExplainer(model, X_train_[:size])
    shap_values = explainer.shap_values(X_train_.iloc[:size].values, check_additivity=False)
    shap_values = clean_deep_shap_values(shap_values, X_train.iloc[:size].shape)
    shap_values_as_df, shap_indices = shap_values_to_df(shap_values, list(X_train.columns))

    get_scores_by_adding_selected_features(X_train, y_train, X_test, y_test, shap_indices,
                                            '../experiments_results/plots/shap/shap 50 samples.png', 'shap 50 samples', model_name)

    size = 100
    print(f"train DeepExplainer on {model_name}")
    explainer = shap.DeepExplainer(model, X_train_[:size])
    shap_values = explainer.shap_values(X_train_.iloc[:size].values, check_additivity=False)
    shap_values = clean_deep_shap_values(shap_values, X_train.iloc[:size].shape)
    shap_values_as_df, shap_indices = shap_values_to_df(shap_values, list(X_train.columns))

    get_scores_by_adding_selected_features(X_train, y_train, X_test, y_test, shap_indices,
                                            '../experiments_results/plots/shap/shap 100 samples.png', 'shap 100 samples', model_name)
