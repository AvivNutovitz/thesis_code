import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import shap
import os
from scipy import stats
import re
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

    elif file_name == 'nasa':
        df.columns = [c.replace(' ', '_') for c in df.columns]
        df.drop(["Neo_Reference_ID", "Name", "Close_Approach_Date", "Epoch_Date_Close_Approach"
                      , "Orbiting_Body", "Orbit_Determination_Date", "Equinox"], axis=1, inplace=True)
        df.Hazardous = [1 if each == True else 0 for each in df.Hazardous]
        y = df.Hazardous
        X = df.drop(["Hazardous"], axis=1)
        return X, y

    else:
        raise ValueError(f"file name can be one of the following: wine, fake_job_posting, hotel_bookings, "
                         f"hr_employee_attrition, nomao, placement_full_class, rain_weather_aus, cervical_cancer, "
                         f"glass or nasa. "
                         f"file_name that passed is {type(file_name)}")


def get_base():
    try:
        return pd.read_csv(os.path.join(os.getcwd(), '..', 'resources/base_36.csv'), header=None)
    except:
        return pd.read_csv(os.path.join(os.getcwd(), 'resources/base_36.csv'), header=None)


def shap_values_to_df(shap_values, feature_names):
    shap_sum = np.abs(shap_values).mean(axis=0)
    if len(shap_sum.shape) > 1:
        shap_sum = shap_sum.mean(axis=0)
    importance_df = pd.DataFrame([feature_names, shap_sum.tolist()]).T
    importance_df.columns = ['feature_name', 'shap_importance']
    importance_df = importance_df.sort_values('shap_importance', ascending=False)
    return importance_df


def t_test_over_doe_shap_differences(shap_values, doe_contributions, feature_names, output_filename='', do_random=False):
    shap_df = shap_values_to_df(shap_values, feature_names)
    if max(shap_df.shap_importance) > 0:
        shap_df.shap_importance = (shap_df.shap_importance/max(shap_df.shap_importance)).apply(lambda x: max(x, 0))
    if not do_random:
        test_df = pd.DataFrame.from_dict(doe_contributions, orient='index').reset_index().rename(columns={
            'index': 'feature_name', 0: 'test_importance'})
        if max(test_df.test_importance) > 0:
            test_df.test_importance = (test_df.test_importance / max(test_df.test_importance)).apply(lambda x: max(x, 0))
    else:
        test_df = pd.DataFrame({'feature_name': feature_names,
                                'test_importance': [random.random() for _ in range(len(feature_names))]})
    full_df = shap_df.set_index('feature_name').join(test_df.set_index('feature_name'))
    if len(output_filename) > 1:
        full_df.to_csv(f't_test_over_doe_shap_differences_{output_filename}.csv')
    return stats.ttest_ind(full_df.shap_importance, full_df.test_importance)

