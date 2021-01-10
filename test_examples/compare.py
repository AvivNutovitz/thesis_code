# --- Imports
from doe_xai import DoeXai
from test_examples import *
from doe_utils import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import shap
from numpy import random
seed = 42

if __name__ == '__main__':
    # ---------------------------
    # ----- base parameters -----
    # ---------------------------
    print("starting...")

    # ----------------------------
    # ----- get all the data -----
    # ----------------------------

    all_data_set_names = ['fake_job_posting', 'rain_weather_aus', 'placement_full_class', 'nomao', 'wine',
                          'hotel_booking', 'hr_employee_attrition', 'cervical_cancer', 'glass', 'mobile_price']

    # ----------------------------------
    # ----- number of replications -----
    # ----------------------------------

    number_of_replications = 3

    # fake_job_posting
    fake_job_posting_X_train, fake_job_posting_y_train, fake_job_posting_X_test, fake_job_posting_y_test = \
        create_fake_job_posting_data_and_tv()
    print("finish load data fake_job_posting")

    # rain_weather_aus
    rain_weather_aus_X_train, rain_weather_aus_y_train, rain_weather_aus_X_test, rain_weather_aus_y_test = \
        create_rain_weather_aus_data()
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

    # cervical_cancer
    cervical_cancer_X_train, cervical_cancer_y_train, cervical_cancer_X_test, cervical_cancer_y_test = \
        create_cervical_cancer_data()
    print("finish load data cervical_cancer")

    # glass
    glass_X_train, glass_y_train, glass_X_test, glass_y_test = create_glass_data()
    print("finish load data glass")

    # mobile_price
    mobile_price_X_train, mobile_price_y_train, mobile_price_X_test, mobile_price_y_test = create_mobile_price_data()
    print("finish load data mobile_price")

    all_X_train = [fake_job_posting_X_train, rain_weather_aus_X_train, placement_full_class_X_train, nomao_X_train,
                   wine_X_train, hotel_booking_X_train, hr_employee_attrition_X_train, cervical_cancer_X_train,
                   glass_X_train, mobile_price_X_train]

    all_X_test = [fake_job_posting_X_test, rain_weather_aus_X_test, placement_full_class_X_test, nomao_X_test,
                  wine_X_test, hotel_booking_X_test, hr_employee_attrition_X_test, cervical_cancer_X_test,
                  glass_X_test, mobile_price_X_test]

    all_y_train = [fake_job_posting_y_train, rain_weather_aus_y_train, placement_full_class_y_train, nomao_y_train,
                   wine_y_train, hotel_booking_y_train, hr_employee_attrition_y_train, cervical_cancer_y_train,
                   glass_y_train, mobile_price_y_train]

    all_y_test = [fake_job_posting_y_test, rain_weather_aus_y_test, placement_full_class_y_test, nomao_y_test,
                  wine_y_test, hotel_booking_y_test, hr_employee_attrition_y_test, cervical_cancer_y_test,
                  glass_y_test, mobile_price_y_test]

    print()

    # ------------------------------
    # ----- Run Experiment ---------
    # ------------------------------

    list_of_models_names = ['LR', 'SVM', 'DTC', 'RF']

    assert len(all_X_train) == len(all_y_train) == len(all_data_set_names) == 10

    for X_train, y_train, data_set_name in zip(all_X_train, all_y_train, all_data_set_names):

        if not os.path.exists(f'../examples_results/{data_set_name}/'):
            os.mkdir(f'../examples_results/{data_set_name}/')

        print(f'working on dataset {data_set_name}')

        # ---------------------------
        # ----- mutual_info ---------
        # ---------------------------

        mutual_info_df = mutual_info_to_df(X_train, y_train)

        # -----------------------
        # ----- f_score ---------
        # -----------------------

        f_score_pvalue_df = f_score_pvalue_to_df(X_train, y_train)

        run_dfs = defaultdict(list)

        for replication_index in range(number_of_replications):
            seed += replication_index
            list_of_models = [LogisticRegression(random_state=random.seed(seed)),
                              SVC(kernel='linear', probability=True, random_state=random.seed(seed)),
                              DecisionTreeClassifier(random_state=random.seed(seed)),
                              RandomForestClassifier(n_estimators=50, random_state=random.seed(seed))]

            print(f'    start replication index {replication_index}')

            for model, model_name in zip(list_of_models, list_of_models_names):

                # ----------------------------
                # ----- build models ---------
                # ----------------------------

                model.fit(X_train, y_train)
                print(f"        finish fit model {model_name}, start Shap at ")

                # ----------------
                # ----- SHAP -----
                # ----------------

                shap_values = None
                if model_name in ['LR', 'SVM']:
                    print(f"        train LinearExplainer on {model_name}")
                    explainer = shap.LinearExplainer(model, X_train)
                    shap_values = explainer.shap_values(X_train)

                elif model_name in ['DTC', 'RF']:
                    print(f"        train TreeExplainer on {model_name}")
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_train)

                shap_values_as_df = shap_values_to_df(shap_values, list(X_train.columns))

                # -------------------
                # ----- DOE XAI -----
                # -------------------

                print(f"        finish Shap, start dfx")
                dx = DoeXai(x_data=X_train, y_data=y_train, model=model)
                cont = dx.find_feature_contribution(only_orig_features=True)
                dfx_importance_as_df = dfx_contribution_to_df(cont)

                # ----------------------------------
                # ----- PERMUTATION IMPORTANCE -----
                # ----------------------------------

                print(f"        finish dfx, start permutation importance")
                permutation_importance_df = permutation_importance_to_df(model, X_train, y_train)

                # ----------------------------------
                # ----- MODEL BASED IMPORTANCE -----
                # ----------------------------------

                print(f"        finish permutation importance, start model feature importance")
                if model_name in ['LR', 'SVM']:
                    model_feature_importance = model.coef_
                elif model_name in ['DTC', 'RF']:
                    model_feature_importance = model.feature_importances_

                model_feature_importance_df = model_feature_importance_to_df(model_feature_importance, X_train.columns)
                print(f"        finish model feature importance, start random importance")

                # -----------------------------
                # ----- RANDOM IMPORTANCE -----
                # -----------------------------

                random_importance_df = random_importance_to_df(X_train.columns)
                print(f"        finish run build importance")

                # build one data set (all models and all 5 tests) per replication
                run_df = pd.concat([shap_values_as_df.set_index('feature_name'),
                                    dfx_importance_as_df.set_index('feature_name'),
                                    permutation_importance_df.set_index('feature_name'),
                                    model_feature_importance_df.set_index('feature_name'),
                                    random_importance_df.set_index('feature_name')], axis=1)

                run_df = run_df.reset_index()
                run_df = run_df.rename(columns={'index': 'feature_name'})
                run_dfs[model_name].append(run_df)

                # output file per replication
                run_df.to_csv(f'../examples_results/{data_set_name}/results_model_{model_name}_replication_{replication_index}.csv', index=False)

                print(f"        finish model {model_name}")
                print()

            print(f'    finish replication index {replication_index}')
            print()

        # -------------------------------------------------------------------------------------------------------
        # ----- RUN t test, kendalltau, pearson, spearman per average feature importance across replication -----
        # -------------------------------------------------------------------------------------------------------

        print(f'start run tests on data set: {data_set_name}')
        print()

        for model_name in list_of_models_names:
            t_dfs = run_dfs[model_name]

            # dfx vs random
            res1 = run_4_tests_on_list_of_dfs(t_dfs, 'dfx_feature_importance', 'random_feature_importance')

            # dfx vs model feature importance
            res2 = run_4_tests_on_list_of_dfs(t_dfs, 'dfx_feature_importance', 'model_feature_importance')

            # dfx vs permutation feature importance
            res3 = run_4_tests_on_list_of_dfs(t_dfs, 'dfx_feature_importance', 'permutation_feature_importance')

            # dfx vs shap
            res4 = run_4_tests_on_list_of_dfs(t_dfs, 'dfx_feature_importance', 'shap_feature_importance')

            pd.concat([res1, res2, res3, res4], axis=1).to_csv(f'../examples_results/{data_set_name}/stats_results_on_model_{model_name}.csv')

        # build one data set (all models and all 4 tests across replications) per data set
        one_df_per_data_set = create_one_metric_df_per_data_set(run_dfs, list_of_models_names)
        one_df_per_data_set.index = list(X_train.columns)

        # avg over models dfx vs mutual_info
        res5 = run_4_tests(one_df_per_data_set['dfx_feature_importance_mean'], set_data_for_statistical_tests(mutual_info_df['mutual_info_score']), 'dfx_feature_importance_mean', 'mutual_info_score')

        # avg over models dfx vs f_score
        res6 = run_4_tests(one_df_per_data_set['dfx_feature_importance_mean'], set_data_for_statistical_tests(f_score_pvalue_df['f_score_pvalue']), 'dfx_feature_importance_mean', 'f_score_pvalue')

        # avg over models dfx vs random_feature_importance
        res7 = run_4_tests(one_df_per_data_set['dfx_feature_importance_mean'], one_df_per_data_set['random_feature_importance_mean'], 'dfx_feature_importance_mean', 'random_feature_importance_mean')

        # avg over models dfx vs model_feature_importance
        res8 = run_4_tests(one_df_per_data_set['dfx_feature_importance_mean'], one_df_per_data_set['model_feature_importance_mean'], 'dfx_feature_importance_mean', 'model_feature_importance_mean')

        # avg over models dfx vs permutation_feature_importance
        res9 = run_4_tests(one_df_per_data_set['dfx_feature_importance_mean'], one_df_per_data_set['permutation_feature_importance_mean'], 'dfx_feature_importance_mean', 'permutation_feature_importance_mean')

        # avg over models dfx vs shap_feature_importance
        res10 = run_4_tests(one_df_per_data_set['dfx_feature_importance_mean'], one_df_per_data_set['shap_feature_importance_mean'], 'dfx_feature_importance_mean', 'shap_feature_importance_mean')

        # output file per data set
        output = pd.concat([res5, res6, res7, res8, res9, res10], axis=1)
        output = output.fillna(-1)
        output.to_csv(f'../examples_results/{data_set_name}/final_stats_on_{data_set_name}.csv')
        one_df_per_data_set.to_csv(f'../examples_results/{data_set_name}/final_feature_importance_on_{data_set_name}.csv')

        print(f'finish run tests on data set: {data_set_name}')
        print('------------------------------------------------')
        print()
