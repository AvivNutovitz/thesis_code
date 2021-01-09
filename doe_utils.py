import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import shap
import os
from scipy import stats
import lime
import lime.lime_tabular
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import f_classif, mutual_info_classif
import random
seed = 42
random.seed(seed)


def create_output_means_of_classes_file(input_file_name, labeled_df):
    df = pd.read_csv(input_file_name)
    global_means = df.T.mean()
    class_columns_means = defaultdict(list)
    yt = pd.DataFrame(labeled_df)
    for gs_column in list(df.columns):
        for cls in list(set(labeled_df)):
            if cls == yt.iloc[int(gs_column)][0]:
                class_columns_means[cls].append(df[gs_column])
    class_means = pd.concat([pd.DataFrame(list_of_lists).mean() for clss, list_of_lists in class_columns_means.items()],
                            axis=1)
    return pd.concat([pd.DataFrame(global_means, columns=['global']), class_means], axis=1)


def create_output_contribution_file(class_feature_contributions, file_name=None):
    if not file_name:
        return pd.DataFrame.from_dict(class_feature_contributions, orient='index').T
    else:
        pd.DataFrame.from_dict(class_feature_contributions, orient='index').T.to_csv(file_name, index=False)


def plot_feature_contributions_like_shap_values_from_df(df, image, to_plot=True):
    feature_contributions_like_shap_values = []
    for col in df.columns:
        feature_contributions_like_shap_values.append(np.nan_to_num(np.array(df[[col]])).reshape((1, 32, 32, 3)))

    if to_plot:
        shap.image_plot(feature_contributions_like_shap_values, image)

    return feature_contributions_like_shap_values


def plot_feature_contributions_like_shap_values(class_feature_contributions, image, to_plot=True):
    feature_contributions_like_shap_values = []
    for key, values in class_feature_contributions.items():
        feature_contributions_like_shap_values.append(np.nan_to_num(np.array(values)).reshape((1, 32, 32, 3)))

    if to_plot:
        shap.image_plot(feature_contributions_like_shap_values, image)

    return feature_contributions_like_shap_values


def plot_feature_histograms(feature_index_histograms, feature_index):
    labels, data = feature_index_histograms[feature_index].keys(), feature_index_histograms[feature_index].values()
    plt.boxplot(data)
    plt.xticks(range(1, len(labels) + 1), labels)
    plt.show()


def load_data(file_name, size=-1):
    example_path = 'example_data'
    df = pd.read_csv(os.path.join(os.getcwd(), '..', f'{example_path}/{file_name}_data.csv'))
    if size > -1:
        df = df.iloc[0:size]

    if file_name == 'wine':
        y = df['y']
        df = df.drop(columns=['y'])
        return df, y

    elif file_name == 'fake_job_posting':
        df.fillna(" ", inplace=True)
        df['text'] = df['title'] + ' ' + df['location'] + ' ' + df['department'] + ' ' + df['company_profile'] + ' ' + \
                     df['description'] + ' ' + df['requirements'] + ' ' + df['benefits'] + ' ' + df['employment_type'] \
                     + ' ' + df['required_education'] + ' ' + df['industry'] + ' ' + df['function']
        return df['text'], df['fraudulent']

    elif file_name == 'hotel_bookings':
        X = df.drop(["is_canceled"], axis=1)
        y = df["is_canceled"]
        return X, y

    elif file_name == 'hr_employee_attrition':
        target_map = {'Yes': 1, 'No': 0}
        # Use the pandas apply method to numerically encode our attrition target variable
        y = df["Attrition"].apply(lambda x: target_map[x])
        X = df.drop(["Attrition"], axis=1)
        return X, y

    elif file_name == 'nomao':
        X = df.drop(["__TARGET__"], axis=1)
        y = df["__TARGET__"]
        return X, y

    elif file_name == 'placement_full_class':
        X = df[['gender', 'ssc_p', 'ssc_b', 'hsc_p', 'hsc_b', 'hsc_s', 'degree_p', 'degree_t', 'workex', 'etest_p',
                'specialisation', 'mba_p']]
        y = df['status']
        return X, y

    elif file_name == 'rain_weather_aus':
        df = df.drop(columns=['Sunshine', 'Evaporation', 'Cloud3pm', 'Cloud9am', 'Location', 'RISK_MM', 'Date'], axis=1)
        df = df.dropna(how='any')
        X = df.loc[:, df.columns != 'RainTomorrow']
        y = df['RainTomorrow']
        return X, y

    elif file_name == 'cervical_cancer':
        df = df.replace('?', np.nan)
        df = df.rename(columns={'Biopsy': 'Cancer'})
        df = df.apply(pd.to_numeric)
        df = df.fillna(df.mean().to_dict())
        X = df.drop('Cancer', axis=1)
        y = df['Cancer']
        return X, y

    elif file_name == 'glass':
        features = df.columns[:-1].tolist()
        X = df[features]
        y = df['Type']
        return X, y

    elif file_name == 'mobile_price':
        y = df.price_range
        X = df.drop(["price_range"], axis=1)
        return X, y

    else:
        raise ValueError(f"file name can be one of the following: wine, fake_job_posting, hotel_bookings, "
                         f"hr_employee_attrition, nomao, placement_full_class, rain_weather_aus, cervical_cancer, "
                         f"glass or mobile_price. "
                         f"file_name that passed is {type(file_name)}")


def get_base():
    try:
        return pd.read_csv(os.path.join(os.getcwd(), '..', 'resources/pb36.csv')) # header=None
    except:
        return pd.read_csv(os.path.join(os.getcwd(), 'resources/pb36.csv')) # header=None


def random_importance_to_df(feature_names):
    return  pd.DataFrame({'feature_name': feature_names,
                          'random_feature_importance': [random.random() for _ in range(len(feature_names))]})


def dfx_contribution_to_df(contribution):
    dfx_df = pd.DataFrame.from_dict(contribution, orient='index')
    dfx_df = dfx_df.reset_index()
    dfx_df.columns = ['feature_name', 'dfx_feature_importance']
    dfx_df = dfx_df.sort_values('dfx_feature_importance', ascending=False)
    return dfx_df


def shap_values_to_df(shap_values, feature_names):
    shap_sum = np.abs(shap_values).mean(axis=0)
    if len(shap_sum.shape) > 1:
        shap_sum = shap_sum.mean(axis=0)
    importance_df = pd.DataFrame([feature_names, shap_sum.tolist()]).T
    importance_df.columns = ['feature_name', 'shap_feature_importance']
    importance_df = importance_df.sort_values('shap_feature_importance', ascending=False)
    return importance_df


def model_feature_importance_to_df(model_feature_importance, feature_names):
    if len(model_feature_importance.shape) > 1:
        model_feature_importance = np.abs(model_feature_importance).mean(axis=0)
    tmp = pd.DataFrame({feature_name: [feature_importance] for feature_name, feature_importance in
                        zip(feature_names, model_feature_importance)}).T
    tmp = tmp.reset_index()
    tmp.columns = ['feature_name', 'model_feature_importance']
    tmp = tmp.sort_values('model_feature_importance', ascending=False)
    return tmp


def permutation_importance_to_df(model, X, y):
    results = permutation_importance(model, X, y, scoring='neg_mean_squared_error')
    feature_importances = results.importances_mean
    tmp = pd.DataFrame({feature_name: [feature_importance] for feature_name, feature_importance in
                         zip(X.columns, feature_importances)}).T
    tmp = tmp.reset_index()
    tmp.columns = ['feature_name', 'permutation_feature_importance']
    tmp = tmp.sort_values('permutation_feature_importance', ascending=False)
    return tmp


def lime_global_importance_to_df(model, X_train, y_train, num_explain=100):
    # Creating the Lime Explainer
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.values,
        training_labels=y_train.values,
        feature_names=X_train.columns.tolist(),
        discretize_continuous=True,
        discretizer="entropy",
    )

    importances = {feature_name: 0.0 for feature_name in X_train.columns}

    # number of instances to generate explanations for
    for i in range(num_explain):
        exp = lime_explainer.explain_instance(X_train.iloc[i],
                                              model.predict_proba,
                                              num_features=X_train.shape[1],
                                              )
        exp_map = exp.as_map()

        # get all feature labels of class index
        feat = [exp_map[1][m][0] for m in range(len(exp_map[1]))]
        # get all feature weights of class index
        weight = [exp_map[1][m][1] for m in range(len(exp_map[1]))]

        # sum the weights, for each feature individually
        for m in range(len(feat)):
            importances[list(X_train.columns)[m]] = importances[list(X_train.columns)[m]] + weight[m]

            # normalize the distribution
    for i in range(X_train.shape[1]):
        importances[list(X_train.columns)[i]] = np.abs(importances[list(X_train.columns)[i]] / (num_explain * 1.0))

    lime_df = pd.DataFrame.from_dict(importances, orient='index').reset_index()
    lime_df.columns = ['feature_name', 'lime_feature_importance']
    lime_df = lime_df.sort_values('lime_feature_importance', ascending=False)
    return lime_df


def f_score_pvalue_to_df(X_train, y_train):
    fs, pvalues = f_classif(X_train, y_train)
    f_pvalue_df = pd.DataFrame({'feature_name': X_train.columns, 'f_score_pvalue': pvalues})
    f_pvalue_df['f_score_pvalue'] = f_pvalue_df['f_score_pvalue'].fillna(-1)
    f_pvalue_df = f_pvalue_df.sort_values('f_score_pvalue', ascending=True)
    return f_pvalue_df


def mutual_info_to_df(X_train, y_train):
    mutual_info = mutual_info_classif(X_train, y_train)
    mutual_info_df = pd.DataFrame({'feature_name': X_train.columns, 'mutual_info_score': mutual_info})
    mutual_info_df['mutual_info_score'] = mutual_info_df['mutual_info_score'].fillna(-1)
    mutual_info_df = mutual_info_df.sort_values('mutual_info_score', ascending=False)
    return mutual_info_df


def set_data_for_statistical_tests(df):
    try:
        # df
        df = df.set_index('feature_name')
        col = df.columns[0]
        if max(df[col]) > 0:
            df[col] = df.values / max(df.values)
    except:
        # series
        if max(df) > 0:
            df = df.values / max(df.values)
            df = pd.Series(df)
    return df


def create_col_mean_from_dfs(dfs, col):
    return pd.concat([df[col] for df in dfs], axis=1).mean(axis=1)


def run_4_tests(t1, t2, col1, col2):
    t_stat, t_pvalue = stats.ttest_ind(t1, t2)
    r_stat, r_pvalue = stats.pearsonr(t1, t2)
    s_stat, s_pvalue = stats.spearmanr(t1, t2)
    k_stat, k_pvalue = stats.kendalltau(t1, t2)
    return pd.DataFrame({f'{col1}_vs_{col2}_stats': [t_stat, r_stat, s_stat, k_stat],
                         f'{col1}_vs_{col2}_pvalue': [t_pvalue, r_pvalue, s_pvalue, k_pvalue]},
                        index=['ttest', 'pearson', 'spearman', 'kendalltau'])


def run_4_tests_on_list_of_dfs(dfs, first_col, second_col):
    t1 = set_data_for_statistical_tests(create_col_mean_from_dfs(dfs, first_col))
    t2 = set_data_for_statistical_tests(create_col_mean_from_dfs(dfs, second_col))
    return run_4_tests(t1, t2, first_col, second_col)


def create_one_metric_df_per_data_set(dfs, list_of_models_names):
    dfx_dfs = []
    permutation_feature_importance_dfs = []
    model_feature_importance_dfs = []
    shap_feature_importance_dfs = []
    random_feature_importance_dfs = []
    for model_name in list_of_models_names:
        t_dfs = dfs[model_name]
        dfx_dfs.append(set_data_for_statistical_tests(create_col_mean_from_dfs(t_dfs, 'dfx_feature_importance')))
        permutation_feature_importance_dfs.append(set_data_for_statistical_tests(create_col_mean_from_dfs(t_dfs, 'permutation_feature_importance')))
        model_feature_importance_dfs.append(set_data_for_statistical_tests(create_col_mean_from_dfs(t_dfs, 'model_feature_importance')))
        shap_feature_importance_dfs.append(set_data_for_statistical_tests(create_col_mean_from_dfs(t_dfs, 'shap_feature_importance')))
        random_feature_importance_dfs.append(set_data_for_statistical_tests(create_col_mean_from_dfs(t_dfs, 'random_feature_importance')))

    all_data = pd.DataFrame({'dfx_feature_importance_mean': list(pd.concat(dfx_dfs, axis=1).mean(axis=1)),
                             'permutation_feature_importance_mean': list(pd.concat(permutation_feature_importance_dfs, axis=1).mean(axis=1)),
                             'model_feature_importance_mean': list(pd.concat(model_feature_importance_dfs, axis=1).mean(axis=1)),
                             'shap_feature_importance_mean': list(pd.concat(shap_feature_importance_dfs, axis=1).mean(axis=1)),
                             'random_feature_importance_mean': list(pd.concat(random_feature_importance_dfs, axis=1).mean(axis=1))})
    return all_data
