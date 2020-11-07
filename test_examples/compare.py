# --- Imports
from doe_xai import DoeXai
from test_examples import *
from doe_utils import shap_values_to_df, t_test_over_doe_shap_differences

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import shap
import scipy.stats as stats

if __name__ == '__main__':
    # ---------------------------
    # ----- base parameters -----
    # ---------------------------

    OUTPUT_FILES = True
    final_results_pvalues = {}
    final_results_kendalltau = {}
    final_results_pvalues_top_5 = {}
    final_results_kendalltau_top_5 = {}
    final_results_t_test_dfx_vs_shap_pvalues = {}
    final_results_t_test_dfx_vs_random_pvalues = {}

    # ----------------------------
    # ----- get all the data -----
    # ----------------------------

    all_data_set_names = ['fake_job_posting', 'rain_weather_aus', 'placement_full_class', 'nomao', 'wine',
                          'hotel_booking', 'hr_employee_attrition']

    # fake_job_posting
    fake_job_posting_X_train, fake_job_posting_y_train, fake_job_posting_X_test, fake_job_posting_y_test = \
        create_fake_job_posting_data_and_tv()
    print("finish load data fake_job_posting")

    # rain_weather_aus
    rain_weather_aus_X_train, rain_weather_aus_y_train, rain_weather_aus_X_test, rain_weather_aus_y_test = \
        create_rain_weather_aus()
    print("finish load data rain_weather_aus")

    # placement_full_class
    placement_full_class_X_train, placement_full_class_y_train, placement_full_class_X_test, \
    placement_full_class_y_test = create_placement_full_class_data()
    print("finish load data placement_full_class")

    # nomao
    nomao_X_train, nomao_y_train, nomao_X_test, nomao_y_test = create_nomao_data()
    print("finish load data nomao")

    # wine
    wine_X_train, wine_y_train, wine_X_test, wine_y_test = create_wine_data()
    print("finish load data wine")

    # hotel_booking
    hotel_booking_X_train, hotel_booking_y_train, hotel_booking_X_test, hotel_booking_y_test = \
        create_hotel_booking_data()
    print("finish load data hotel_booking")

    # hr_employee_attrition
    hr_employee_attrition_X_train, hr_employee_attrition_y_train, hr_employee_attrition_X_test, \
    hr_employee_attrition_y_test = create_hr_employee_attrition_data()
    print("finish load data hr_employee_attrition")

    all_X_train = [fake_job_posting_X_train, rain_weather_aus_X_train, placement_full_class_X_train, nomao_X_train,
                   wine_X_train, hotel_booking_X_train, hr_employee_attrition_X_train]

    all_X_test = [fake_job_posting_X_test, rain_weather_aus_X_test, placement_full_class_X_test, nomao_X_test,
                   wine_X_test, hotel_booking_X_test, hr_employee_attrition_X_test]

    all_y_train = [fake_job_posting_y_train, rain_weather_aus_y_train, placement_full_class_y_train, nomao_y_train,
                   wine_y_train, hotel_booking_y_train, hr_employee_attrition_y_train]

    all_y_test = [fake_job_posting_y_test, rain_weather_aus_y_test, placement_full_class_y_test, nomao_y_test,
                   wine_y_test, hotel_booking_y_test, hr_employee_attrition_y_test]

    # ------------------------------
    # ----- Run Experiment ---------
    # ------------------------------

    list_of_models = [LogisticRegression(),
                      MultinomialNB(),
                      SVC(kernel='linear', probability=True),
                      DecisionTreeClassifier(),
                      RandomForestClassifier(n_estimators=50)]
    list_of_models_names = ['LR', 'MNB', 'SVM', 'DTC', 'RF']

    assert len(all_X_train) == len(all_y_train) == len(all_data_set_names) == 7

    for X_train, y_train, data_set_name in zip(all_X_train, all_y_train, all_data_set_names):

        print(f'working on dataset {data_set_name}')
        dataset_results_kendalltau = []
        dataset_results_pvalues = []
        dataset_results_kendalltau_top_5 = []
        dataset_results_pvalues_top_5 = []
        dataset_results_t_test_dfx_vs_shap_pvalues = []
        dataset_results_t_test_dfx_vs_random_pvalues = []

        for model, model_name in zip(list_of_models, list_of_models_names):

            # ----------------------------
            # ----- build models ---------
            # ----------------------------

            model.fit(X_train, y_train)
            print(f"    finish fit model {model_name}")

            # ----------------
            # ----- Shap -----
            # ----------------

            shap_values = None
            if model_name in ['LR', 'MNB', 'SVM']:
                print(f"    train LinearExplainer on {model_name}")
                explainer = shap.LinearExplainer(model, X_train)
                shap_values = explainer.shap_values(X_train)

            elif model_name in ['DTC', 'RF']:
                print(f"    train TreeExplainer on {model_name}")
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_train)

            shap_values_as_df = shap_values_to_df(shap_values, list(X_train.columns))
            shap_values_as_df = shap_values_as_df.sort_values(by='shap_importance', ascending=False)

            # -------------------
            # ----- DOE XAI -----
            # -------------------

            print(f"    finish shap, start DOE XAI")
            dx = DoeXai(x_data=X_train, y_data=y_train, model=model)
            cont = dx.find_feature_contribution(only_orig_features=True)
            df_doe_importance = pd.DataFrame.from_dict(cont, orient='index').reset_index().\
                rename(columns={'index': 'feature_name', 0: "doe_importance"})
            df_doe_importance = df_doe_importance.sort_values(by='doe_importance', ascending=False)

            # ---------------------------
            # ----- kendalltau test -----
            # ---------------------------

            print(f"    run kendalltau_test on shap VS DOE XAI")
            assert df_doe_importance.shape == shap_values_as_df.shape
            kendalltau, p_value = stats.kendalltau(shap_values_as_df[['feature_name']],
                                                   df_doe_importance[['feature_name']])
            dataset_results_pvalues.append(p_value)
            dataset_results_kendalltau.append(kendalltau)
            print(kendalltau, p_value)
            print()

            print(f"    run kendalltau_test on shap VS DOE XAI on top 5 features")
            assert df_doe_importance.head().shape == shap_values_as_df.head().shape
            kendalltau_top_5, p_value_top_5 = stats.kendalltau(shap_values_as_df.head()[['feature_name']],
                                                   df_doe_importance.head()[['feature_name']])
            dataset_results_pvalues_top_5.append(p_value_top_5)
            dataset_results_kendalltau_top_5.append(kendalltau_top_5)
            print(kendalltau_top_5, p_value_top_5)
            print()

            # ------------------
            # ----- t test -----
            # ------------------

            cont = dx.find_feature_contribution(only_orig_features=True)
            _, pvalue_vs_shap = t_test_over_doe_shap_differences(shap_values, cont, X_train.columns, do_random=False)
            dataset_results_t_test_dfx_vs_shap_pvalues.append(pvalue_vs_shap)
            print(f"    run t_test over doe and shap differences")
            print(pvalue_vs_shap)

            _, pvalue_vs_random = t_test_over_doe_shap_differences(shap_values, cont, X_train.columns, do_random=True)
            dataset_results_t_test_dfx_vs_random_pvalues.append(pvalue_vs_random)
            print(f"    run t_test over doe and random differences")
            print(pvalue_vs_random)
            print()

        final_results_pvalues[data_set_name] = dataset_results_pvalues
        final_results_kendalltau[data_set_name] = dataset_results_kendalltau

        final_results_pvalues_top_5[data_set_name] = dataset_results_pvalues_top_5
        final_results_kendalltau_top_5[data_set_name] = dataset_results_kendalltau_top_5

        final_results_t_test_dfx_vs_shap_pvalues[data_set_name] = dataset_results_t_test_dfx_vs_shap_pvalues
        final_results_t_test_dfx_vs_random_pvalues[data_set_name] = dataset_results_t_test_dfx_vs_random_pvalues

    final_results_pvalues_df = pd.DataFrame.from_dict(final_results_pvalues, orient='index',
                                                      columns=list_of_models_names)
    final_results_kendalltau_df = pd.DataFrame.from_dict(final_results_kendalltau, orient='index',
                                                         columns=list_of_models_names)

    final_results_pvalues_df_top_5 = pd.DataFrame.from_dict(final_results_pvalues_top_5, orient='index',
                                                      columns=list_of_models_names)
    final_results_kendalltau_df_top_5 = pd.DataFrame.from_dict(final_results_kendalltau_top_5, orient='index',
                                                         columns=list_of_models_names)

    final_results_t_test_dfx_vs_shap_pvalues_df = pd.DataFrame.from_dict(final_results_t_test_dfx_vs_shap_pvalues,
                                                                         orient='index', columns=list_of_models_names)
    final_results_t_test_dfx_vs_random_pvalues_df = pd.DataFrame.from_dict(final_results_t_test_dfx_vs_random_pvalues,
                                                                           orient='index', columns=list_of_models_names)

    if OUTPUT_FILES:
        final_results_pvalues_df.to_csv('final_results_pvalues_df.csv', index=False)
        final_results_kendalltau_df.to_csv('final_results_kendalltau_df.csv', index=False)

        final_results_pvalues_df_top_5.to_csv('final_results_pvalues_df_top_5.csv', index=False)
        final_results_kendalltau_df_top_5.to_csv('final_results_kendalltau_df_top_5.csv', index=False)

        final_results_t_test_dfx_vs_shap_pvalues_df.to_csv('final_results_t_test_dfx_vs_shap_pvalues_df.csv',
                                                           index=False)
        final_results_t_test_dfx_vs_random_pvalues_df.to_csv('final_results_t_test_dfx_vs_random_pvalues_df.csv',
                                                           index=False)
