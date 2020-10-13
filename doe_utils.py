import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import shap
import os


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


def create_output_contribution_file(class_feature_contributions, global_feature_contributions):
    df_output = pd.DataFrame({key: np.abs(value[0]) for key, value in class_feature_contributions.items()})
    df_output.columns = ['class_{}'.format(col) for col in list(df_output.columns)]
    df_output['global'] = pd.DataFrame({key: np.abs(value[0]) for key, value in global_feature_contributions.items()})
    return df_output


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


def get_base():
    try:
        return pd.read_csv(os.path.join(os.getcwd(), '..', 'resources/base_36.csv'), header=None)
    except:
        return pd.read_csv(os.path.join(os.getcwd(), 'resources/base_36.csv'), header=None)

# todo
# def get_feature_names_combinations(feature_names):
#     new_feature_names = []
#     list_of_columns_pairs = list(itertools.combinations(feature_names, 2))
#     for pair in list_of_columns_pairs:
#         new_feature_names.append(str(pair[0]) + '-' + str(pair[1]))
#     return new_feature_names
